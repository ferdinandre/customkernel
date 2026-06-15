"""
Tunables + MetricLog: the data the live panel binds to.

* ``Tunables`` is a thread-safe store of declared hyperparameters, each with
  optional slider metadata (lo/hi/log). The training loop reads ``tn.lr`` each
  iteration; the panel and ``%pset`` write to it. Every write is timestamped in
  ``.history`` so the panel can drop "lr 3e-4 -> 1e-4" markers on the curve.

* ``MetricLog`` is a thread-safe ring buffer the loop appends to
  (``m.log(step=.., reward=.., kl=..)``). The panel reads downsampled series
  for the curve and the latest values for the diagnostic cards. The loop only
  ever touches plain Python objects here -- never the widget -- which keeps
  high-frequency logging off the comm channel.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class Param:
    """A declared tunable's value + slider metadata."""

    __slots__ = ("value", "lo", "hi", "log", "kind")

    def __init__(self, value, lo=None, hi=None, log=False):
        self.value = value
        self.kind = type(value).__name__
        if lo is None or hi is None:
            # Sensible default range around the initial value.
            if isinstance(value, bool):
                lo, hi, log = 0, 1, False
            elif value == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = value / 100 if value > 0 else value * 100
                hi = value * 100 if value > 0 else value / 100
        self.lo, self.hi, self.log = lo, hi, bool(log)

    def as_dict(self, name: str) -> Dict[str, Any]:
        return {"name": name, "value": self.value, "lo": self.lo,
                "hi": self.hi, "log": self.log, "kind": self.kind}


def _coerce(spec) -> Param:
    if isinstance(spec, Param):
        return spec
    if isinstance(spec, tuple):
        # (value,), (value, lo, hi), (value, lo, hi, "log")
        value = spec[0]
        lo = spec[1] if len(spec) > 1 else None
        hi = spec[2] if len(spec) > 2 else None
        log = len(spec) > 3 and spec[3] in ("log", True)
        return Param(value, lo, hi, log)
    return Param(spec)


class Tunables:
    """Declared, observable hyperparameters.

    Tunables(lr=(3e-4, 1e-5, 1e-2, "log"), entropy=(0.01, 0.0, 0.1),
             use_gae=True)
    """

    def __init__(self, **specs):
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_params",
                           {k: _coerce(v) for k, v in specs.items()})
        object.__setattr__(self, "_history", [])   # (t, name, old, new)
        object.__setattr__(self, "_version", 0)     # bumps on external edits

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            try:
                return self._params[name].value
            except KeyError:
                raise AttributeError(f"no tunable {name!r}") from None

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set(name, value, external=True)

    def _set(self, name, value, external: bool):
        with self._lock:
            p = self._params.get(name)
            if p is None:
                self._params[name] = _coerce(value)
                old = "<new>"
            else:
                old = p.value
                # keep the declared type
                try:
                    value = type(p.value)(value)
                except (TypeError, ValueError):
                    pass
                p.value = value
            self._history.append((time.time(), name, old, value))
            if external:
                self._version += 1

    __getitem__ = __getattr__
    __setitem__ = __setattr__

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: p.value for k, p in self._params.items()}

    def controls(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [p.as_dict(k) for k, p in self._params.items()]

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def history(self) -> List[Tuple[float, str, Any, Any]]:
        with self._lock:
            return list(self._history)

    def panel(self, metrics: "Optional[MetricLog]" = None, **kw):
        """Convenience: build the live panel bound to these tunables."""
        from .panel import Panel
        return Panel(tunables=self, metrics=metrics, **kw)

    def __repr__(self):
        return "Tunables(" + ", ".join(
            f"{k}={p.value!r}" for k, p in self.snapshot().items()) + ")"


class MetricLog:
    """Thread-safe metric buffer the training loop appends to."""

    def __init__(self, max_points: int = 5000):
        self._lock = threading.RLock()
        self._rows: Deque[Tuple[float, Optional[int], Dict[str, float]]] = \
            deque(maxlen=max_points)
        self._keys: List[str] = []

    def log(self, step: Optional[int] = None, **values) -> None:
        with self._lock:
            self._rows.append((time.time(), step, dict(values)))
            for k in values:
                if k not in self._keys:
                    self._keys.append(k)

    # convenience: m(step, reward=...) also works
    __call__ = log

    @property
    def keys(self) -> List[str]:
        with self._lock:
            return list(self._keys)

    def latest(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._rows[-1][2]) if self._rows else {}

    @property
    def last_step(self) -> Optional[int]:
        with self._lock:
            return self._rows[-1][1] if self._rows else None

    def series(self, key: str, max_points: int = 300, tail: Optional[int] = None
               ) -> List[Tuple[float, float]]:
        """Downsampled [(x, value)] for `key`; x is step if present else idx.
        If `tail` is set, only the last `tail` matching points are considered
        (a recent window) before downsampling."""
        with self._lock:
            pts = [(i if s is None else s, row.get(key))
                   for i, (t, s, row) in enumerate(self._rows)
                   if key in row]
        if tail:
            pts = pts[-tail:]
        if len(pts) <= max_points:
            return pts
        stride = len(pts) / max_points
        return [pts[int(i * stride)] for i in range(max_points)]

    def times(self) -> List[Tuple[float, Optional[int]]]:
        """(wall_time, step) for each row -- used to place change markers."""
        with self._lock:
            return [(t, s) for (t, s, _) in self._rows]
