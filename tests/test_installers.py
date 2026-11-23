"""Consistency checks for the one-step installers (no network needed)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PS1 = ROOT / "install_windows_gpu.ps1"
SH = ROOT / "install_mac_linux.sh"
LOCK = ROOT / "requirements" / "dcm-lock.txt"
TORCH = ROOT / "requirements" / "torch.txt"


def _ps1_var(name):
    m = re.search(rf'^\${name}\s*=\s*"([^"]+)"', PS1.read_text(encoding="ascii"), re.M)
    assert m, name
    return m.group(1)


def _sh_var(name):
    m = re.search(rf'^{name}="?([^"\n]+)"?$', SH.read_text(encoding="utf-8"), re.M)
    assert m, name
    return m.group(1)


def test_installers_pin_the_same_versions():
    pins = dict(line.split("==") for line in TORCH.read_text().split())
    assert _ps1_var("TorchVersion") == _sh_var("TORCH_VERSION") == pins["torch"]
    assert _ps1_var("TorchvisionVersion") == _sh_var("TORCHVISION_VERSION") == pins["torchvision"]
    assert _ps1_var("PythonVersion") == _sh_var("PYTHON_VERSION") == "3.12"
    assert _ps1_var("SapiensUrl") == _sh_var("SAPIENS_URL")
    assert _ps1_var("UvVersion") == _sh_var("UV_VERSION")
    assert _ps1_var("SapiensUrl") in (ROOT / "requirements" / "dcm.in").read_text()


def test_lock_leaves_torch_to_the_gpu_specific_step():
    pinned = {line.split("==")[0].lower() for line in LOCK.read_text().splitlines()
              if line and not line.startswith("#")}
    assert not pinned & {"torch", "torchvision", "triton", "myogait", "sapiens"}
    assert not any(p.startswith("nvidia-") for p in pinned)
    for needed in ("numpy", "scipy", "pandas", "gaitkit", "streamlit", "mediapipe", "safetensors"):
        assert needed in pinned, needed
    assert all("==" in line for line in LOCK.read_text().splitlines() if line and not line.startswith("#"))


def test_windows_scripts_are_ascii_with_crlf():
    # Windows PowerShell 5.1 reads BOM-less scripts as ANSI: non-ASCII bytes can turn into quotes.
    for name in ("install_windows_gpu.ps1", "install_windows_gpu.bat", "run_dashboard.bat", "update_windows.bat"):
        raw = (ROOT / name).read_bytes()
        raw.decode("ascii")
        assert b"\n" not in raw.replace(b"\r\n", b""), f"{name} must use CRLF line endings"


def test_launchers_point_at_existing_files():
    assert "dcm_dashboard.py" in (ROOT / "run_dashboard.bat").read_text()
    assert (ROOT / "dcm_dashboard.py").exists() and (ROOT / "scripts" / "check_gpu.py").exists()
    assert "install_windows_gpu.ps1" in (ROOT / "install_windows_gpu.bat").read_text()
    assert "requirements\\dcm-lock.txt" in PS1.read_text(encoding="ascii")
