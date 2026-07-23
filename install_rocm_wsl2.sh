#!/bin/bash
# =============================================================================
# AMD ROCm + PyTorch 训练环境一键安装脚本 (WSL2 / Ubuntu 24.04)
# =============================================================================
# 适配显卡: AMD Radeon RX 9070 XT (RDNA 4 / gfx1201)
# 使用方法:
#   1. 在 Windows 11 上启用 WSL2 并安装 Ubuntu 24.04
#   2. 将本脚本复制到 WSL2 的 ~/ 目录
#   3. chmod +x install_rocm.sh && bash install_rocm.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

echo "============================================================"
echo "  AMD ROCm + PyTorch 训练环境安装"
echo "  目标: RX 9070 XT (RDNA 4) on WSL2 Ubuntu 24.04"
echo "============================================================"
echo ""

# ---- 0. Check WSL2 & OS ----
if ! grep -qi "microsoft" /proc/version 2>/dev/null; then
    err "请在 WSL2 中运行本脚本"
fi

OS_VER=$(lsb_release -rs 2>/dev/null || echo "0")
if [ "$(echo "$OS_VER < 24.04" | bc -l 2>/dev/null || echo 1)" = "1" ]; then
    warn "推荐 Ubuntu 24.04，当前版本: $OS_VER"
fi

log "WSL2 Ubuntu $OS_VER 确认"

# ---- 1. System packages ----
log "安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    wget gnupg2 curl software-properties-common \
    python3 python3-pip python3-venv \
    build-essential cmake pkg-config \
    libstdc++-12-dev \
    2>&1 | tail -1

# ---- 2. AMD ROCm 6.3 repository ----
log "配置 AMD ROCm 6.3 仓库..."
if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
    wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - \
        | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.3 noble main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    sudo apt-get update -qq
    log "ROCm 仓库已添加"
else
    log "ROCm 仓库已存在，跳过"
fi

# ---- 3. Install ROCm runtime (minimal for PyTorch) ----
log "安装 ROCm 运行时库 (仅 PyTorch 必需的组件)..."
sudo apt-get install -y -qq \
    rocm-libs \
    rccl \
    hipblas hipblaslt \
    hipsparse \
    hipfft \
    hipsolver \
    rocrand rocblas \
    miopen-hip \
    2>&1 | tail -3

# Add user to render group
sudo usermod -a -G render,video "$USER" 2>/dev/null || true
log "ROCm 运行时安装完成"

# ---- 4. Python 虚拟环境 ----
log "创建 Python 虚拟环境..."
PROJ_DIR="/mnt/d/work files/imageidentification"
VENV_DIR="$HOME/rocm_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log "虚拟环境创建: $VENV_DIR"
else
    log "虚拟环境已存在"
fi

source "$VENV_DIR/bin/activate"

# ---- 5. PyTorch with ROCm ----
log "安装 PyTorch + torchvision (ROCm 6.3)..."
pip install --upgrade pip -q
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.3 \
    2>&1 | tail -5

# ---- 6. Python dependencies ----
log "安装项目依赖..."
if [ -f "$PROJ_DIR/requirements.txt" ]; then
    pip install -r "$PROJ_DIR/requirements.txt" 2>&1 | tail -5
else
    warn "未找到 requirements.txt，跳过"
fi

# ---- 7. Verify ----
echo ""
echo "============================================================"
echo "  验证安装"
echo "============================================================"

python3 -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA/ROCm:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'VRAM:         {vram:.1f} GB')
    print(f'Device count: {torch.cuda.device_count()}')
" || warn "PyTorch GPU 检测失败，可能需要重启 WSL2"

echo ""
echo "============================================================"
echo "  安装完成!"
echo "============================================================"
echo ""
echo "  使用方法:"
echo "    1. 激活环境: source $VENV_DIR/bin/activate"
echo "    2. 开始训练: python train_launch.py --data dataset.yaml"
echo ""
echo "  如果 GPU 未检测到，运行:"
echo "    sudo tee /etc/ld.so.conf.d/rocm.conf <<< /opt/rocm/lib"
echo "    sudo ldconfig"
echo "    然后重启 WSL2: wsl --shutdown (在 Windows PowerShell 中)"
echo ""
