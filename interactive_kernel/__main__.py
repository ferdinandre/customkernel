"""Launch the interactive kernel: python -m interactive_kernel -f <conn-file>"""
from ipykernel.kernelapp import IPKernelApp
from .kernel import InteractiveKernel

IPKernelApp.launch_instance(kernel_class=InteractiveKernel)
