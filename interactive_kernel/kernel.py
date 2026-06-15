"""
InteractiveKernel: an ipykernel subclass where every cell is pausable.

Differences from stock ipykernel:

* ``execute_request`` code runs on a dedicated worker thread that is
  registered with the PauseEngine. The shell channel still processes cells
  sequentially (normal notebook semantics), but the *control* channel --
  which ipykernel >= 6 services on its own thread -- stays responsive, so a
  frontend can pause/inspect/mutate/resume MID-CELL via the custom messages
  below. No %%bg needed.

* Custom control-channel messages (msg_type -> reply msg_type):
    ik_pause_request         -> ik_pause_reply        {ok, stack}
    ik_resume_request        -> ik_resume_reply       {ok}
    ik_stop_request          -> ik_stop_reply         {ok}
    ik_status_request        -> ik_status_reply       {state, ...}
    ik_inspect_request       -> ik_inspect_reply      {stack, locals}
                                content: {depth: int = 0}
    ik_set_variable_request  -> ik_set_variable_reply {ok, error?}
                                content: {name, value_expr, depth = 0}
  ``value_expr`` is evaluated in the user namespace, so frontends can send
  ``"1e-4"`` or ``"np.linspace(0, 1, 5)"`` alike.

* ``interrupt_request`` (kernelspec uses ``interrupt_mode: message``) is
  rerouted through the engine: instead of SIGINT (which can't reach the
  worker thread), StopExecution is raised at the next line of user code --
  a *clean*, catchable interruption.

The IPython magics layer is auto-loaded, so %%bg etc. also work here.
"""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

from ipykernel.ipkernel import IPythonKernel

from .engine import default_engine
from .magics import load_ipython_extension


def _safe_repr(value, limit: int = 200) -> str:
    try:
        r = repr(value)
    except Exception as e:                                   # noqa: BLE001
        r = f"<unrepresentable: {e}>"
    return r if len(r) <= limit else r[:limit] + "..."


class InteractiveKernel(IPythonKernel):
    implementation = "interactive_kernel"
    implementation_version = "0.1.0"
    banner = "Python with pausable execution (interactive_kernel)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = default_engine
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ik-exec",
            initializer=self.engine.register_current_thread,
        )
        for msg_type in ("ik_pause_request", "ik_resume_request",
                         "ik_stop_request", "ik_status_request",
                         "ik_inspect_request", "ik_set_variable_request"):
            self.control_handlers[msg_type] = getattr(self, msg_type)
        # Clean interrupts via the engine instead of SIGINT.
        self.control_handlers["interrupt_request"] = self.ik_interrupt_request
        load_ipython_extension(self.shell)

    # ------------------------------------------------------------------ #
    # Execution: every cell runs on the registered worker thread
    # ------------------------------------------------------------------ #

    async def do_execute(self, code, silent, store_history=True,
                         user_expressions=None, allow_stdin=False,
                         **kwargs):
        shell = self.shell
        loop = asyncio.get_running_loop()

        def run():
            return shell.run_cell(code, store_history=store_history,
                                  silent=silent)

        try:
            result = await loop.run_in_executor(self._executor, run)
        except BaseException:                                # noqa: BLE001
            traceback.print_exc()
            return {"status": "error", "execution_count": shell.execution_count,
                    "ename": "KernelExecutionError", "evalue": "", "traceback": []}

        err = result.error_before_exec or result.error_in_exec
        if err is None:
            reply = {"status": "ok",
                     "user_expressions":
                         shell.user_expressions(user_expressions or {})}
        else:
            reply = {"status": "error",
                     "ename": type(err).__name__,
                     "evalue": str(err),
                     "traceback": getattr(result, "traceback", []) or []}
        reply["execution_count"] = shell.execution_count - 1
        reply["payload"] = shell.payload_manager.read_payload()
        shell.payload_manager.clear_payload()
        return reply

    # ------------------------------------------------------------------ #
    # Custom control-channel handlers
    # ------------------------------------------------------------------ #

    def _reply(self, stream, ident, parent, msg_type, content):
        self.session.send(stream, msg_type, content, parent, ident=ident)

    async def ik_pause_request(self, stream, ident, parent):
        timeout = float(parent.get("content", {}).get("timeout", 5.0))
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None, lambda: self.engine.pause(timeout=timeout))
        content = {"ok": ok, "stack": self.engine.stack() if ok else []}
        self._reply(stream, ident, parent, "ik_pause_reply", content)

    async def ik_resume_request(self, stream, ident, parent):
        self.engine.resume()
        self._reply(stream, ident, parent, "ik_resume_reply", {"ok": True})

    async def ik_stop_request(self, stream, ident, parent):
        self.engine.stop()
        self._reply(stream, ident, parent, "ik_stop_reply", {"ok": True})

    async def ik_interrupt_request(self, stream, ident, parent):
        self.engine.stop()
        self._reply(stream, ident, parent, "interrupt_reply", {"status": "ok"})

    async def ik_status_request(self, stream, ident, parent):
        self._reply(stream, ident, parent, "ik_status_reply",
                    self.engine.status())

    async def ik_inspect_request(self, stream, ident, parent):
        depth = int(parent.get("content", {}).get("depth", 0))
        try:
            content = {"ok": True,
                       "stack": self.engine.stack(),
                       "locals": {k: _safe_repr(v)
                                  for k, v in self.engine.locals(depth).items()}}
        except Exception as e:                               # noqa: BLE001
            content = {"ok": False, "error": str(e)}
        self._reply(stream, ident, parent, "ik_inspect_reply", content)

    async def ik_set_variable_request(self, stream, ident, parent):
        c = parent.get("content", {})
        name, expr = c.get("name"), c.get("value_expr")
        depth = int(c.get("depth", 0))
        try:
            value = eval(expr, self.shell.user_ns)           # noqa: S307
            self.engine.set_local(name, value, depth)
            content = {"ok": True, "name": name, "value": _safe_repr(value)}
        except Exception as e:                               # noqa: BLE001
            content = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._reply(stream, ident, parent, "ik_set_variable_reply", content)
