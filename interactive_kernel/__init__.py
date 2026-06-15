"""Pausable execution for Jupyter: custom kernel + Colab-compatible extension.

Public API:
    from interactive_kernel import Tunables, MetricLog, checkpoint
    %load_ext interactive_kernel        # gives %%bg, %pause, %pset, ... + ictl
    panel = ictl.panel(tn, m)           # live anywidget control panel
"""
from .engine import PauseEngine, StopExecution, default_engine, checkpoint
from .tunables import Tunables, MetricLog, Param
from .magics import load_ipython_extension

__version__ = "0.2.0"
__all__ = ["PauseEngine", "StopExecution", "default_engine", "checkpoint",
           "Tunables", "MetricLog", "Param", "load_ipython_extension"]

# --- Colab: auto-enable third-party widgets so the panel renders with no
# --- user incantation. No-op everywhere else.
try:                                    # pragma: no cover
    import google.colab  # noqa: F401
    from google.colab import output as _colab_output
    try:
        _colab_output.enable_custom_widget_manager()
    except Exception:
        pass
except ImportError:
    pass
