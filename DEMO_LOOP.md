# A demo loop that actually responds to steering

The quickstart loop saturates (reward -> 100 and stays there), so tweaking
hyperparams does nothing visible. This one is sensitive to both knobs:

- `lr` controls how fast reward chases a moving target (slope / responsiveness)
- `entropy` is exploration noise added to reward (visible jitter)

Set entropy to 0 -> the band of noise collapses to a smooth line.
Raise lr -> reward tracks the target faster. Drop lr -> it lags.

```python
from interactive_kernel import Tunables, MetricLog, checkpoint
import random, math

tn = Tunables(
    lr=(0.05, 1e-3, 0.5, "log"),
    entropy=(0.3, 0.0, 2.0),
)
m = MetricLog()
```

```python
%%bg
import time, random, math
reward = 0.0
step = 0
while step < 1_000_000:
    step += 1
    checkpoint()
    target = 100 + 30 * math.sin(step / 400)      # a moving goal
    reward += tn.lr * (target - reward)            # chase it at rate lr
    observed = reward + random.gauss(0, tn.entropy * 10)   # + exploration noise
    m.log(step=step, reward=observed, lr=tn.lr, entropy=tn.entropy)
    time.sleep(0.02)
```

```python
# window=400 shows only recent steps, so steering effects fill the view
# instead of being squished against the whole-run history
panel = ictl.panel(tn, m, window=400, ylabel="episode reward")
panel
```

Try: drag `entropy` to 0 and watch the jitter vanish; drag `lr` up and watch
reward snap to the sine target; drop `lr` low and watch it lag behind. Each
change drops a labelled marker on the curve.
