"""Report the installed PyTorch build and whether it really runs on the GPU.

Used by the installers. Prints one JSON line; exit code 0 means a CUDA
kernel ran correctly on the GPU, 1 means no usable GPU, 2 means PyTorch is
not installed.
"""

import json
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - only hit on broken installs
    print(json.dumps({"torch": None, "error": str(exc)}))
    sys.exit(2)

info = {
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "kernel_test": False,
}
if info["cuda_available"]:
    try:
        x = torch.arange(16, device="cuda", dtype=torch.float32)
        ok_math = float((x * 2).sum().item()) == 240.0
        conv = torch.nn.functional.conv2d(
            torch.ones(1, 3, 8, 8, device="cuda"), torch.ones(4, 3, 3, 3, device="cuda"))
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=torch.cuda.get_device_name(0),
            vram_gb=round(props.total_memory / 1024 ** 3, 1),
            compute_capability=f"{props.major}.{props.minor}",
            kernel_test=bool(ok_math and conv.shape[1] == 4),
        )
    except Exception as exc:  # e.g. "no kernel image is available for this device"
        info["error"] = str(exc)

print(json.dumps(info))
sys.exit(0 if info["kernel_test"] else 1)
