#!/usr/bin/env bash
# Verify that PyTorch can see the AMD iGPU (gfx1151) through ROCm.
# Acceptance: torch.cuda.is_available() is True and the device name prints.
set -euo pipefail

# gfx1151 is not in ROCm's official support list; this override makes HIP treat
# it as the supported gfx1150 ISA. Expected to be set persistently on the host,
# but we default it here so the script is self-contained.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "ROCM_PATH=$ROCM_PATH"
echo

python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not importable. Install the ROCm wheel (see requirements.txt).")

print("torch:", torch.__version__)
ok = torch.cuda.is_available()
print("cuda available (ROCm HIP):", ok)
if not ok:
    sys.exit("ERROR: no GPU visible to torch. Check ROCm install and HSA_OVERRIDE_GFX_VERSION.")

print("device:", torch.cuda.get_device_name(0))

# Tiny on-device op to confirm compute actually runs, not just enumeration.
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum().item()
print("matmul smoke test ok, sum=%.3f" % y)
print("\nROCm verification PASSED")
PY
