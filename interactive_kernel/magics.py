"""
IPython extension layer. Load with ``%load_ext interactive_kernel``.

This is the part that works *everywhere* a stock IPython kernel runs --
including Google Colab's wrapper kernel, where custom kernels can't be
installed but extensions load fine.

Magics
------
%%bg                 run this cell on a background, pausable thread
%pause [timeout]     pause the running background cell at its next line
%resume              resume
%pstop               stop the background cell (raises StopExecution in it)
%pstack              show the paused call stack
%plocals [depth]     show locals of the paused frame (or a caller via depth)
%pset name expr [d]  set a local in the paused frame, e.g.  %pset lr 1e-4
%pstatus             engine status

A programmatic facade is also injected into the user namespace as ``ictl``:
ictl.pause(), ictl.resume(), ictl.stop(), ictl.locals(), ictl.stack(),
ictl.set_local(name, value, depth=0).
"""

from __future__ import annotations

import linecache
import threading
import traceback
from typing import Any, Dict, Optional

from .engine import default_engine, StopExecution

_bg_counter = 0
_bg_jobs: Dict[int, threading.Thread] = {}


class Controls:
    """Thin user-facing facade over the engine (injected as ``ictl``)."""

    def __init__(self, engine):
        self._e = engine

    def pause(self, timeout: float = 5.0) -> bool:
        ok = self._e.pause(timeout=timeout)
        if ok:
            top = self._e.stack()[0]
            print(f"paused in {top['function']}() at "
                  f"{top['file']}:{top['line']}")
        else:
            print("nothing reached a pause point (is a %%bg cell running?)")
        return ok

    def resume(self) -> None:
        self._e.resume()
        print("resumed")

    def stop(self) -> None:
        self._e.stop()
        print("stop requested (takes effect at next line of user code)")

    def stack(self):
        return self._e.stack()

    def locals(self, depth: int = 0) -> Dict[str, Any]:
        return self._e.locals(depth)

    def set_local(self, name: str, value: Any, depth: int = 0) -> None:
        self._e.set_local(name, value, depth)
        print(f"set {name} = {value!r} (depth {depth})")

    def status(self):
        return self._e.status()


def _register_magics(ipython, engine) -> None:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic

    @magics_class
    class InteractiveMagics(Magics):

        @cell_magic
        def bg(self, line, cell):
            """Run the cell body on a background, pausable thread."""
            global _bg_counter
            _bg_counter += 1
            fname = f"<bg-cell-{_bg_counter}>"
            # Make the source visible to tracebacks / inspect.
            linecache.cache[fname] = (len(cell), None,
                                      cell.splitlines(True), fname)
            code = compile(cell, fname, "exec")
            ns = self.shell.user_ns

            def job():
                try:
                    exec(code, ns)
                except StopExecution:
                    print(f"[bg {_bg_counter}] stopped")
                except BaseException:
                    traceback.print_exc()

            t = engine.run_in_thread(job, name=fname)
            _bg_jobs[_bg_counter] = t
            print(f"[bg {_bg_counter}] started on background thread -- "
                  f"use %pause / %pset / %resume / %pstop")

        @line_magic
        def pause(self, line):
            timeout = float(line) if line.strip() else 5.0
            ipython.user_ns["ictl"].pause(timeout=timeout)

        @line_magic
        def resume(self, line):
            ipython.user_ns["ictl"].resume()

        @line_magic
        def pstop(self, line):
            ipython.user_ns["ictl"].stop()

        @line_magic
        def pstack(self, line):
            for f in engine.stack():
                marker = "*" if f["is_user_code"] else " "
                print(f"{marker} [{f['depth']}] {f['function']}()  "
                      f"{f['file']}:{f['line']}")

        @line_magic
        def plocals(self, line):
            depth = int(line) if line.strip() else 0
            for k, v in engine.locals(depth).items():
                r = repr(v)
                print(f"  {k} = {r[:120] + '...' if len(r) > 120 else r}")

        @line_magic
        def pset(self, line):
            parts = line.split(None, 2)
            if len(parts) < 2:
                print("usage: %pset NAME EXPR [DEPTH]"); return
            name, expr = parts[0], parts[1]
            depth = 0
            if len(parts) == 3:
                try:
                    depth = int(parts[2])
                except ValueError:
                    expr = parts[1] + " " + parts[2]
            value = eval(expr, ipython.user_ns)          # noqa: S307
            engine.set_local(name, value, depth)
            print(f"set {name} = {value!r} (depth {depth})")

        @line_magic
        def pstatus(self, line):
            print(engine.status())

    ipython.register_magics(InteractiveMagics)


def load_ipython_extension(ipython) -> None:
    """Entry point for ``%load_ext interactive_kernel``."""
    engine = default_engine
    ipython.user_ns["ictl"] = Controls(engine)
    _register_magics(ipython, engine)
    print("interactive_kernel loaded: %%bg, %pause, %pset, %resume, %pstop, "
          "%pstack, %plocals + `ictl` API")
