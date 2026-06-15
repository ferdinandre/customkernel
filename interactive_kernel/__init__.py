"""Pausable execution for Jupyter: custom kernel + Colab-compatible extension."""
from .engine import PauseEngine, StopExecution, default_engine
from .magics import load_ipython_extension

__version__ = "0.1.0"
__all__ = ["PauseEngine", "StopExecution", "default_engine",
           "load_ipython_extension"]
