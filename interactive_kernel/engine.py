"""
Core pause engine. No Jupyter dependencies -- this works in any CPython.

Strategy
--------
* Python >= 3.12: ``sys.monitoring`` (PEP 669). Zero overhead while running
  (no events enabled). When a pause/stop is requested we enable LINE events,
  the next line of *user code* on a registered thread parks (or raises
  KeyboardInterrupt for stop), and events are switched off again on resume.
* Python < 3.12: ``sys.settrace`` fallback with a cheap call-event-only
  global tracer, installed on worker threads we own.

Frame mutation uses ``frame.f_locals`` write-through on 3.13+ (PEP 667) and
the classic ``PyFrame_LocalsToFast`` ctypes flush on older CPython.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

_HAS_MONITORING = hasattr(sys, "monitoring")


class StopExecution(KeyboardInterrupt):
    """Raised inside user code when a stop was requested."""


def write_frame_local(frame, name: str, value: Any) -> None:
    if sys.version_info >= (3, 13):
        frame.f_locals[name] = value
        return
    frame.f_locals[name] = value
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame),
                                          ctypes.c_int(1))


def _default_user_code_filter(filename: str) -> bool:
    """Heuristic: is this code the user's (a notebook cell / their script)?"""
    if "interactive_kernel" in filename or "threading.py" in filename:
        return False
    if filename.startswith("<ipython-input") or filename.startswith("<bg-cell"):
        return True
    if "ipykernel_" in filename and filename.endswith(".py"):
        return True   # modern IPython compiles cells to /tmp/ipykernel_x/*.py
    if "<stdin>" in filename or "<string>" in filename:
        return True
    return not ("site-packages" in filename
                or "lib/python" in filename
                or filename.startswith("<frozen"))


class PauseEngine:
    """Preemptively pause/inspect/mutate/stop user code on registered threads.

    States: idle -> armed -> paused -> idle (on resume).
    """

    _MON_TOOL_NAME = "interactive-kernel"

    def __init__(self,
                 user_code_filter: Callable[[str], bool] = _default_user_code_filter):
        self._filter = user_code_filter
        self._lock = threading.RLock()
        self._threads: set[int] = set()          # idents of pausable threads
        self._resume_evt = threading.Event(); self._resume_evt.set()
        self._paused_evt = threading.Event()
        self._stop_requested = False
        self._paused_frame = None
        self._paused_thread: Optional[int] = None
        self._pending_edits: Dict[str, Any] = {}
        self._mon_tool: Optional[int] = None
        self.state = "idle"

    # ------------------------------------------------------------------ #
    # Thread registration
    # ------------------------------------------------------------------ #

    def register_current_thread(self) -> None:
        with self._lock:
            self._threads.add(threading.get_ident())
        if not _HAS_MONITORING:
            sys.settrace(self._legacy_global_trace)

    def unregister_current_thread(self) -> None:
        with self._lock:
            self._threads.discard(threading.get_ident())
        if not _HAS_MONITORING:
            sys.settrace(None)

    def run_in_thread(self, fn: Callable, *args: Any,
                      name: str = "ik-worker", **kwargs: Any) -> threading.Thread:
        """Run fn on a new registered (pausable) thread."""
        def worker():
            self.register_current_thread()
            try:
                fn(*args, **kwargs)
            except StopExecution:
                pass
            except BaseException:
                traceback.print_exc()
            finally:
                self.unregister_current_thread()
        t = threading.Thread(target=worker, name=name, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------ #
    # Control surface
    # ------------------------------------------------------------------ #

    def pause(self, timeout: Optional[float] = 5.0) -> bool:
        """Arm pausing; returns True once some user thread is parked."""
        with self._lock:
            if self.state == "paused":
                return True
            self._paused_evt.clear()
            self._resume_evt.clear()
            self.state = "armed"
            self._arm()
        if timeout is None:
            return True
        ok = self._paused_evt.wait(timeout)
        if not ok:
            with self._lock:
                if self.state == "armed":   # nothing hit a pause point
                    self.state = "idle"
                    self._resume_evt.set()
                    self._disarm()
        return ok

    def resume(self) -> None:
        with self._lock:
            self._resume_evt.set()

    def stop(self) -> None:
        """Raise StopExecution at the next user-code line on any registered
        thread (works whether running or paused)."""
        with self._lock:
            self._stop_requested = True
            self._resume_evt.set()      # release a paused thread first
            self._arm()                 # ensure events are on to catch it

    # -- paused-frame inspection / surgery ------------------------------- #

    def frame_at(self, depth: int = 0):
        frame = self._paused_frame
        if frame is None:
            raise RuntimeError("no thread is paused")
        for _ in range(depth):
            if frame.f_back is None:
                raise IndexError("stack not that deep")
            frame = frame.f_back
        return frame

    def locals(self, depth: int = 0) -> Dict[str, Any]:
        return dict(self.frame_at(depth).f_locals)

    def stack(self) -> List[Dict[str, Any]]:
        frame, out, depth = self._paused_frame, [], 0
        while frame is not None:
            out.append({"depth": depth,
                        "function": frame.f_code.co_name,
                        "file": frame.f_code.co_filename,
                        "line": frame.f_lineno,
                        "is_user_code": self._filter(frame.f_code.co_filename)})
            frame, depth = frame.f_back, depth + 1
        return out

    def set_local(self, name: str, value: Any, depth: int = 0) -> None:
        write_frame_local(self.frame_at(depth), name, value)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"state": self.state,
                    "registered_threads": len(self._threads),
                    "paused_thread": self._paused_thread}

    # ------------------------------------------------------------------ #
    # The actual park (both backends funnel here)
    # ------------------------------------------------------------------ #

    def _maybe_park(self, frame) -> None:
        if self._stop_requested:
            self._stop_requested = False
            self._disarm()
            self.state = "idle"
            raise StopExecution("stopped by user")
        if self._resume_evt.is_set():
            return
        with self._lock:
            self._paused_frame = frame
            self._paused_thread = threading.get_ident()
            self.state = "paused"
        self._paused_evt.set()
        try:
            self._resume_evt.wait()
        finally:
            with self._lock:
                self._paused_frame = None
                self._paused_thread = None
                if self.state == "paused":
                    self.state = "idle"
                self._disarm()
        if self._stop_requested:
            self._stop_requested = False
            raise StopExecution("stopped by user")

    # ------------------------------------------------------------------ #
    # Backend: sys.monitoring (3.12+)
    # ------------------------------------------------------------------ #

    def _acquire_tool(self) -> Optional[int]:
        if self._mon_tool is not None:
            return self._mon_tool
        mon = sys.monitoring
        for tool in (mon.DEBUGGER_ID, mon.OPTIMIZER_ID, mon.PROFILER_ID):
            try:
                if mon.get_tool(tool) is None:
                    mon.use_tool_id(tool, self._MON_TOOL_NAME)
                    mon.register_callback(tool, mon.events.LINE, self._mon_line)
                    self._mon_tool = tool
                    return tool
            except ValueError:
                continue
        return None

    def _arm(self) -> None:
        if not _HAS_MONITORING:
            # settrace only fires on *new* calls; attach to frames that are
            # already running (this is how debugger attach works).
            with self._lock:
                idents = set(self._threads)
            for ident, frame in sys._current_frames().items():
                if ident not in idents:
                    continue
                f = frame
                while f is not None:
                    if self._filter(f.f_code.co_filename):
                        f.f_trace = self._legacy_line_trace
                        f.f_trace_lines = True
                    f = f.f_back
            return
        tool = self._acquire_tool()
        if tool is None:
            raise RuntimeError("no free sys.monitoring tool id (debugger "
                               "attached?) -- cannot arm preemptive pause")
        sys.monitoring.restart_events()  # re-enable previously DISABLEd sites
        sys.monitoring.set_events(tool, sys.monitoring.events.LINE)

    def _disarm(self) -> None:
        if _HAS_MONITORING and self._mon_tool is not None:
            sys.monitoring.set_events(self._mon_tool, 0)

    def _mon_line(self, code, line_number):
        if threading.get_ident() not in self._threads:
            if not self._filter(code.co_filename):
                return sys.monitoring.DISABLE   # never interesting
            return None
        if not self._filter(code.co_filename):
            return sys.monitoring.DISABLE
        # Depth 1 = the frame executing the monitored line.
        self._maybe_park(sys._getframe(1))
        return None

    # ------------------------------------------------------------------ #
    # Backend: sys.settrace fallback (< 3.12)
    # ------------------------------------------------------------------ #

    def _legacy_global_trace(self, frame, event, arg):
        if event != "call":
            return None
        if self._resume_evt.is_set() and not self._stop_requested:
            return None
        if not self._filter(frame.f_code.co_filename):
            return None
        return self._legacy_line_trace

    def _legacy_line_trace(self, frame, event, arg):
        if event == "line":
            self._maybe_park(frame)
            if self._resume_evt.is_set() and not self._stop_requested:
                return None
        return self._legacy_line_trace


# A process-wide default engine (what the kernel and the Colab plugin share).
default_engine = PauseEngine()
