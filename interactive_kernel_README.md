# interactive_kernel

A Jupyter kernel where running cells can be **paused**, their live variables
**inspected and edited**, and execution **resumed** — built for steering long
deep-RL training runs without restarting them.

```
%%bg
while step < total_steps:          # training keeps state: optimizer, buffers, env
    ...

%pause            → paused in train() at <bg-cell-1>:7
%plocals          → step = 18432, lr = 0.0003, ...
%pset lr 1e-4
%resume
```

## What's in the box

| File | What it is |
|---|---|
| `interactive_kernel/engine.py` | Core pause engine. `sys.monitoring` (PEP 669) on Python ≥ 3.12 — zero overhead while running — with a `settrace` + frame-attach fallback for older Pythons. Frame edits via PEP 667 on 3.13+, `PyFrame_LocalsToFast` before. No Jupyter dependencies. |
| `interactive_kernel/magics.py` | IPython extension: `%%bg`, `%pause`, `%pset`, `%resume`, `%pstop`, `%pstack`, `%plocals`, `%pstatus`, plus the `ictl` Python API. Works on **any** stock IPython kernel, including Colab. |
| `interactive_kernel/kernel.py` | The custom kernel: every cell runs on a pausable worker thread; custom `ik_*` control-channel messages let a frontend pause/inspect/mutate/resume **mid-cell** with no `%%bg`; interrupt is rerouted through the engine as a clean, catchable `StopExecution`. |
| `install.py` | Registers the kernelspec ("Python 3 (interactive)"). |
| `ikctl.py` | Terminal client: control a running kernel out-of-band (`python ikctl.py pause`, `set lr 1e-4`, `resume`). This is also the reference for how your future frontend talks to the kernel. |

## Test it in JupyterLab (5 minutes)

```bash
pip install -e ".[kernel]"        # from this directory
python install.py --user
jupyter lab
```

Pick the **Python 3 (interactive)** kernel. Two ways to drive it:

**A. From the notebook (works in any frontend):** put `%%bg` at the top of
your training cell, then run `%pause`, `%pset lr 1e-4`, `%resume` from other
cells.

**B. Mid-cell, no `%%bg` (the custom-kernel superpower):** run any
long-running cell normally, then from a terminal:

```bash
python ikctl.py pause
python ikctl.py stack
python ikctl.py locals
python ikctl.py set lr 1e-4
python ikctl.py resume
```

The control channel is serviced on its own thread, so this works while the
cell is still "running" in the UI. Pressing the stop button also goes through
the engine now — it raises a clean `StopExecution` at the next line instead
of an unpredictable SIGINT.

## Google Colab

Colab **force-starts its own wrapper kernel** and ignores installed
kernelspecs, so the custom kernel and `ikctl.py` won't work there. The
extension layer does:

```python
!pip install git+https://github.com/you/interactive-kernel   # or upload the folder
%load_ext interactive_kernel
```

Then use workflow A (`%%bg` + magics). Same engine, same frame editing —
you just lose the mid-cell control channel because Colab owns the kernel.
This also works in stock Jupyter if you don't want to switch kernels.

## Writing steerable training loops

- **`while` beats `for`.** Editing the loop variable of `for step in
  range(N)` does nothing — the iterator overwrites it next pass. Write
  `while step < total:` and the loop re-reads what you edit.
- **Edit what gets re-read.** A paused-frame edit only matters if later
  iterations read that variable. An `lr` already baked into an optimizer
  must be pushed into `optimizer.param_groups` inside the loop — keep a
  `for g in opt.param_groups: g["lr"] = lr` line in there.
- **GIL-bound loops:** pure-Python busy loops without any I/O or native
  calls can starve the control thread; almost any real DRL loop (torch ops,
  env steps) yields constantly, so this rarely matters in practice.

## Known limitations (MVP)

- Output from `%%bg` cells attaches to whichever cell most recently
  executed — a known Jupyter behavior for background threads. The custom
  frontend will fix this by tagging iopub messages.
- `shell.run_cell` off the main thread: history and most features work, but
  libraries that install signal handlers (some env wrappers do) will
  complain — that only works on main threads. Wrap or disable those.
- The engine grabs a `sys.monitoring` tool id (debugger slot first). If the
  JupyterLab debugger (debugpy) is attached, slots may be taken — the engine
  falls back through optimizer/profiler ids, but don't run both at once.
- One pause target at a time: the first registered thread to hit a line
  event wins. Multi-run support is a planned protocol extension.
