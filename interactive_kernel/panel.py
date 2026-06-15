"""
Live training panel as an anywidget -- renders inline in Jupyter, JupyterLab,
Colab, and VS Code from one code path, no build step.

Design that keeps it smooth on Colab:
* The training loop only ever touches plain Python (Tunables, MetricLog).
* A single daemon "pusher" thread owned by the panel snapshots that state a
  few times a second and writes traitlets -> one throttled comm stream,
  instead of the loop spamming the comm channel.
* Slider edits flow JS -> Python via custom messages and write the Tunables
  store directly; the pusher only re-pushes control *definitions* when the
  store's version changes (an external %pset), so a user dragging a slider
  isn't fought by an echo from Python.

If anywidget isn't installed, Panel() raises a clear install hint.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from .engine import default_engine
from .tunables import MetricLog, Tunables

try:
    import anywidget
    import traitlets
    _HAS_ANYWIDGET = True
    _Base = anywidget.AnyWidget
except Exception:                                   # noqa: BLE001
    _HAS_ANYWIDGET = False

    class _Base:                                    # placeholder
        pass


_ESM = r"""
function fmt(v) {
  if (typeof v !== "number") return String(v);
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5))
    return v.toExponential(2);
  return (Math.round(v * 1e4) / 1e4).toString();
}

function sliderToValue(c, frac) {
  if (c.log) {
    const lo = Math.log10(c.lo), hi = Math.log10(c.hi);
    return Math.pow(10, lo + frac * (hi - lo));
  }
  let v = c.lo + frac * (c.hi - c.lo);
  if (c.kind === "int" || c.kind === "bool") v = Math.round(v);
  return v;
}
function valueToSlider(c) {
  if (c.log) {
    const lo = Math.log10(c.lo), hi = Math.log10(c.hi);
    return (Math.log10(c.value) - lo) / (hi - lo);
  }
  return (c.value - c.lo) / (c.hi - c.lo);
}

function render({ model, el }) {
  el.innerHTML = "";
  const root = document.createElement("div");
  root.style.cssText =
    "font-family: var(--jp-ui-font-family, system-ui, sans-serif);" +
    "border:1px solid rgba(128,128,128,.3); border-radius:12px;" +
    "padding:14px; max-width:680px;";
  el.appendChild(root);

  const banner = document.createElement("div");
  banner.style.cssText =
    "display:flex; align-items:center; justify-content:space-between;" +
    "padding:8px 12px; border-radius:8px; margin-bottom:12px; font-size:14px;";
  const bannerText = document.createElement("span");
  const btns = document.createElement("div");
  const mkBtn = (label, type) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText =
      "margin-left:6px; font-size:13px; padding:4px 12px; cursor:pointer;" +
      "border:1px solid rgba(128,128,128,.4); border-radius:6px;" +
      "background:transparent; color:inherit;";
    b.onclick = () => model.send({ type });
    return b;
  };
  btns.append(mkBtn("pause", "pause"), mkBtn("resume", "resume"),
              mkBtn("stop", "stop"));
  banner.append(bannerText, btns);
  root.appendChild(banner);

  const curveWrap = document.createElement("div");
  curveWrap.style.cssText =
    "position:relative; height:160px; border:1px solid rgba(128,128,128,.2);" +
    "border-radius:8px; padding:8px; margin-bottom:12px;";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.setAttribute("viewBox", "0 0 600 140");
  svg.setAttribute("preserveAspectRatio", "none");
  curveWrap.appendChild(svg);
  const markerLayer = document.createElement("div");
  markerLayer.style.cssText =
    "position:absolute; inset:8px; pointer-events:none;";
  curveWrap.appendChild(markerLayer);
  root.appendChild(curveWrap);

  const slidersWrap = document.createElement("div");
  slidersWrap.style.cssText =
    "display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));" +
    "gap:10px; margin-bottom:12px;";
  root.appendChild(slidersWrap);

  const cardsWrap = document.createElement("div");
  cardsWrap.style.cssText =
    "display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr));" +
    "gap:8px;";
  root.appendChild(cardsWrap);

  function drawBanner() {
    const st = model.get("_state");
    const step = model.get("_step");
    const palette = {
      running: ["rgba(29,158,117,.15)", "#1D9E75"],
      paused: ["rgba(239,159,39,.18)", "#BA7517"],
      armed: ["rgba(239,159,39,.12)", "#BA7517"],
      idle: ["rgba(128,128,128,.12)", "inherit"],
      stopped: ["rgba(128,128,128,.15)", "inherit"],
    }[st] || ["rgba(128,128,128,.12)", "inherit"];
    banner.style.background = palette[0];
    bannerText.style.color = palette[1];
    const label = st === "paused" ? "paused" :
      st === "armed" ? "pause armed" : st;
    bannerText.textContent =
      label + (step != null ? "  ·  step " + step : "");
  }

  function drawCurve() {
    const series = model.get("_series") || [];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    markerLayer.innerHTML = "";
    if (series.length < 2) return;
    const xs = series.map(p => p[0]), ys = series.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = v => (x1 === x0 ? 0 : (v - x0) / (x1 - x0)) * 600;
    const sy = v => 140 - (y1 === y0 ? 0.5 : (v - y0) / (y1 - y0)) * 130 - 5;
    const path = document.createElementNS(svg.namespaceURI, "polyline");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#378ADD");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("points", series.map(p => sx(p[0]) + "," + sy(p[1])).join(" "));
    svg.appendChild(path);

    (model.get("_markers") || []).forEach(m => {
      if (m.x < x0 || m.x > x1) return;
      const frac = (x1 === x0 ? 0 : (m.x - x0) / (x1 - x0));
      const line = document.createElement("div");
      line.style.cssText =
        "position:absolute; top:0; bottom:0; width:0;" +
        "border-left:1.5px dashed rgba(128,128,128,.6); left:" +
        (frac * 100) + "%;";
      const tag = document.createElement("div");
      tag.textContent = m.label;
      tag.style.cssText =
        "position:absolute; top:0; left:" + (frac * 100) + "%;" +
        "font-size:11px; opacity:.75; padding:0 4px; white-space:nowrap;" +
        "transform:translateX(2px);";
      markerLayer.append(line, tag);
    });
  }

  function buildSliders() {
    slidersWrap.innerHTML = "";
    (model.get("_controls") || []).forEach(c => {
      const card = document.createElement("div");
      card.style.cssText =
        "border:1px solid rgba(128,128,128,.25); border-radius:8px;" +
        "padding:10px 12px;";
      const head = document.createElement("div");
      head.style.cssText =
        "display:flex; justify-content:space-between; align-items:baseline;" +
        "margin-bottom:6px; font-size:13px;";
      const name = document.createElement("span");
      name.textContent = c.name;
      name.style.opacity = ".7";
      const val = document.createElement("span");
      val.textContent = fmt(c.value);
      val.style.cssText = "font-weight:500; font-family:monospace;";
      head.append(name, val);

      const input = document.createElement("input");
      input.type = "range";
      input.min = 0; input.max = 1000; input.step = 1;
      input.value = Math.round(valueToSlider(c) * 1000);
      input.style.width = "100%";
      input.oninput = () => {
        const v = sliderToValue(c, input.value / 1000);
        c.value = v;
        val.textContent = fmt(v);
        model.send({ type: "set", name: c.name, value: v });
      };
      card.append(head, input);
      slidersWrap.appendChild(card);
    });
  }

  function drawCards() {
    const latest = model.get("_metrics_latest") || {};
    cardsWrap.innerHTML = "";
    Object.entries(latest).forEach(([k, v]) => {
      const card = document.createElement("div");
      card.style.cssText =
        "background:rgba(128,128,128,.1); border-radius:8px; padding:10px 12px;";
      const lab = document.createElement("div");
      lab.textContent = k; lab.style.cssText = "font-size:12px; opacity:.7;";
      const num = document.createElement("div");
      num.textContent = fmt(v);
      num.style.cssText = "font-size:16px; font-weight:500;";
      card.append(lab, num);
      cardsWrap.appendChild(card);
    });
  }

  model.on("change:_state", drawBanner);
  model.on("change:_step", drawBanner);
  model.on("change:_series", drawCurve);
  model.on("change:_markers", drawCurve);
  model.on("change:_metrics_latest", drawCards);
  model.on("change:_controls_version", buildSliders);

  drawBanner(); drawCurve(); buildSliders(); drawCards();
}
export default { render };
"""


class Panel(_Base):
    if _HAS_ANYWIDGET:
        _esm = _ESM
        _controls = traitlets.List().tag(sync=True)
        _controls_version = traitlets.Int(0).tag(sync=True)
        _series = traitlets.List().tag(sync=True)
        _markers = traitlets.List().tag(sync=True)
        _metrics_latest = traitlets.Dict().tag(sync=True)
        _state = traitlets.Unicode("idle").tag(sync=True)
        _step = traitlets.Int(0, allow_none=True).tag(sync=True)

    def __init__(self, tunables: Optional[Tunables] = None,
                 metrics: Optional[MetricLog] = None,
                 primary: Optional[str] = None,
                 hz: float = 3.0, engine=None, **kw):
        if not _HAS_ANYWIDGET:
            raise ImportError(
                "the live panel needs anywidget: pip install anywidget "
                "(already a dependency of interactive-kernel; run "
                "`pip install -U interactive-kernel`)")
        super().__init__(**kw)
        self._t = tunables or Tunables()
        self._m = metrics or MetricLog()
        self._engine = engine or default_engine
        self._primary = primary
        self._period = 1.0 / max(hz, 0.5)
        self._last_version = -1
        self._stop_pusher = False
        self.on_msg(self._on_msg)
        self._push(initial=True)
        self._pusher = threading.Thread(target=self._loop, daemon=True,
                                        name="ik-panel-pusher")
        self._pusher.start()

    # -- JS -> Python ---------------------------------------------------- #

    def _on_msg(self, _widget, content, _buffers):
        t = content.get("type")
        if t == "set":
            self._t._set(content["name"], content["value"], external=False)
        elif t == "pause":
            self._engine.pause(timeout=0)        # non-blocking arm
        elif t == "resume":
            self._engine.resume()
        elif t == "stop":
            self._engine.stop()

    # -- Python -> JS (throttled) --------------------------------------- #

    def _primary_key(self) -> Optional[str]:
        if self._primary:
            return self._primary
        keys = self._m.keys
        for cand in ("reward", "return", "ep_reward", "loss"):
            if cand in keys:
                return cand
        return keys[0] if keys else None

    def _compute_markers(self) -> List[Dict[str, Any]]:
        times = self._m.times()
        if not times:
            return []
        out = []
        for (t, name, old, new) in self._t.history:
            # place marker at the step of the nearest metric row in time
            best = min(times, key=lambda ts: abs(ts[0] - t))
            x = best[1] if best[1] is not None else times.index(best)
            out.append({"x": x, "label": f"{name} \u2192 "
                        f"{new if not isinstance(new, float) else round(new, 6)}"})
        return out[-12:]

    def _push(self, initial: bool = False) -> None:
        v = self._t.version
        if initial or v != self._last_version:
            self._controls = self._t.controls()
            self._controls_version = v
            self._last_version = v
        key = self._primary_key()
        if key:
            self._series = [[float(x), float(y)]
                            for (x, y) in self._m.series(key)
                            if y is not None]
            self._markers = self._compute_markers()
        self._metrics_latest = {k: float(v) for k, v
                                in self._m.latest().items()}
        self._step = self._m.last_step or 0
        self._state = self._engine.state

    def _loop(self) -> None:
        while not self._stop_pusher:
            try:
                self._push()
            except Exception:                       # noqa: BLE001
                pass
            time.sleep(self._period)

    def close(self):                                # stop the pusher cleanly
        self._stop_pusher = True
        try:
            super().close()
        except Exception:                           # noqa: BLE001
            pass
