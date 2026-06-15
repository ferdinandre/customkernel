# Colab quickstart

## Step 0 — 30-second sanity check (do this once, before trusting it)

Confirm anywidget renders and updates live on *your* current Colab image.
Paste into a fresh cell:

```python
!pip -q install anywidget
import anywidget, traitlets
class Probe(anywidget.AnyWidget):
    _esm = """
    function render({model, el}){
      const s=document.createElement('input'); s.type='range';
      const o=document.createElement('span'); o.textContent=model.get('v');
      s.oninput=()=>{model.set('v',+s.value); model.save_changes(); o.textContent=s.value;};
      model.on('change:v',()=>{o.textContent=model.get('v');});
      el.append(s,o);
    }
    export default {render};
    """
    v = traitlets.Int(5).tag(sync=True)
p = Probe(); p
```

Drag the slider; the number should track it. Then in another cell run `p.v`
— it should reflect the slider, and setting `p.v = 8` should move the slider.
If that works, the live panel will work. (interactive_kernel auto-enables
Colab's custom widget manager on import, so you won't see the "third-party
widgets" warning.)

## Step 1 — install + load

```python
!pip install git+https://github.com/ferdinandre/customkernel
%load_ext interactive_kernel
```

## Step 2 — declare tunables + a metric log

```python
from interactive_kernel import Tunables, MetricLog, checkpoint

tn = Tunables(
    lr=(3e-4, 1e-5, 1e-2, "log"),   # value, lo, hi, log-scale slider
    entropy=(0.01, 0.0, 0.1),
)
m = MetricLog()
```

## Step 3 — train on a background thread, reading tunables each step

```python
%%bg
import time, math
step = 0
while step < 1_000_000:
    step += 1
    checkpoint()                     # instant pause point (optional but nice)
    lr = tn.lr                       # read fresh each iteration
    ent = tn.entropy
    reward = 100*(1-math.exp(-step*lr)) + ent*10
    m.log(step=step, reward=reward, lr=lr)   # plain Python, no widget contact
    time.sleep(0.02)
```

## Step 4 — show the live panel (renders right in the cell)

```python
panel = ictl.panel(tn, m)
panel
```

Drag the `lr` slider while training runs — the change lands on the next
iteration and drops a marker on the reward curve. Use the pause / resume /
stop buttons, or the magics (`%pause`, `%pwait`, `%pset lr 1e-4`, `%resume`).

## Notes
- `checkpoint()` makes pausing instant even if an iteration is a long native
  call. Without it you can still pause, but only at Python line boundaries.
- Update rate is throttled (~3 Hz) so Colab's comm channel stays smooth even
  if you log every step.
- `%pause` is now non-blocking (arms and returns); use `%pwait` to block until
  it actually parks.
