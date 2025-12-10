#!/bin/bash
set -euo pipefail

# 待处理的Python包（每行一个包名）
PYTHON_PACKAGES=(
    "darts"
    "matplotlib"
    "numpy"
    "pandas"
    "scikit_learn"
    "scipy"
    "statsmodels"
    "torch"
    "ray"
    "tqdm"
    "dash"
    "dash-bootstrap-components"
    # "reformer-pytorch"
)

# 核心逻辑：已存在则保留当前版本，不存在则安装最新版（单个包失败不终止整体）
process_package() {
    local pkg_name=$1
    local import_name=$(echo "$pkg_name" | tr '-' '_')  # 适配Python导入格式（如reformer-pytorch→reformer_pytorch）

    # 1. 检查包是否已安装（忽略版本）
    if python3 -c "import $import_name" &>/dev/null; then
        local current_version=$(python3 -m pip show "$pkg_name" | grep "Version" | awk '{print $2}')
        echo "[已存在] $pkg_name - 当前版本：$current_version（保留，不重新安装）"
        return
    fi

    # 2. 包不存在，尝试安装最新版（允许失败）
    echo "[待安装] $pkg_name - 尝试安装最新版"
    if python3 -m pip install "$pkg_name" &>/dev/null; then
        local new_version=$(python3 -m pip show "$pkg_name" | grep "Version" | awk '{print $2}')
        echo "[安装成功] $pkg_name - 已安装最新版本：$new_version"
        return
    fi

    # 3. 安装失败，标记并跳过（不终止脚本）
    echo "[安装失败] $pkg_name 安装失败，跳过该包（建议检查系统依赖或Python版本）"
}

# 检查包是否安装（忽略版本，最终汇总）
check_installation() {
    local pkg_name=$1
    local import_name=$(echo "$pkg_name" | tr '-' '_')

    if python3 -c "import $import_name" &>/dev/null; then
        local installed_version=$(python3 -m pip show "$pkg_name" | grep "Version" | awk '{print $2}')
        echo "[已安装] $pkg_name (实际版本: $installed_version)"
    else
        echo "[未安装] $pkg_name"
    fi
}

# 前置提示：CentOS7系统依赖安装建议
if [ -f /etc/redhat-release ] && grep -q "CentOS Linux 7" /etc/redhat-release; then
    echo "检测到CentOS7系统，建议先安装必要系统依赖（避免编译失败）："
    echo "sudo yum install -y gcc gcc-c++ python3-devel libgfortran openblas-devel"
    read -p "是否已安装系统依赖？[y/N] " -n 1 -r
    echo -e "\n"
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "⚠️  未安装系统依赖，可能导致部分包（如scipy、lightgbm）安装失败！"
        read -p "是否继续执行？[y/N] " -n 1 -r
        echo -e "\n"
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
fi

# 批量处理所有包
echo "=== 开始初始化算力服务器Python环境 ==="
for pkg in "${PYTHON_PACKAGES[@]}"; do
    process_package "$pkg"
done

# 最终安装状态检查（汇总所有包结果）
echo -e "\n=== 最终安装状态汇总（忽略版本，仅判断是否安装） ==="
for pkg in "${PYTHON_PACKAGES[@]}"; do
    check_installation "$pkg"
done

echo -e "\n=== 环境初始化完成 ==="
echo "📌 注意：标记为[未安装]的包，需手动检查系统依赖或Python版本兼容性"