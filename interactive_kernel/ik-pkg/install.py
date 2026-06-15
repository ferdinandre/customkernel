"""Register the kernelspec:  python install.py [--user]"""
import json, os, sys, tempfile
from jupyter_client.kernelspec import KernelSpecManager

spec = {
    "argv": [sys.executable, "-m", "interactive_kernel", "-f", "{connection_file}"],
    "display_name": "Python 3 (interactive)",
    "language": "python",
    "interrupt_mode": "message",
}
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "kernel.json"), "w") as f:
        json.dump(spec, f, indent=2)
    dest = KernelSpecManager().install_kernel_spec(
        d, "interactive_python", user="--user" in sys.argv or None)
    print(f"installed kernelspec to {dest}")
    print("select 'Python 3 (interactive)' in JupyterLab / VS Code")
