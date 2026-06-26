#!/bin/bash
set -e

echo "=== Lambda Cloud Setup Script ==="

# 1. Target User & Directory Setup
TARGET_USER=${SUDO_USER:-$(whoami)}
if [ "$TARGET_USER" = "root" ]; then
    if id "ubuntu" &>/dev/null; then
        TARGET_USER="ubuntu"
    fi
fi
USER_HOME=$(eval echo ~$TARGET_USER)
echo "Target User: $TARGET_USER"
echo "Home Directory: $USER_HOME"

# 2. Package updates (only if run as root/sudo)
if [ "$(id -u)" -eq 0 ]; then
    echo "Updating system packages..."
    apt-get update && apt-get install -y git python3-pip python3-venv python3-dev
fi

# 3. Setup Git Config
echo "Configuring Git..."
# We run git config as target user so the settings go into their home directory ~/.gitconfig
sudo -u "$TARGET_USER" git config --global user.name "gm169"
sudo -u "$TARGET_USER" git config --global user.email "grzesmakosa@gmail.com"

# 4. Clone/Fetch Repository
REPO_DIR="$USER_HOME/nano_gpt_jax"
REPO_URL="https://github.com/gmk561/nano_gpt_jax.git"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository to $REPO_DIR..."
    sudo -u "$TARGET_USER" git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository already exists at $REPO_DIR. Pulling latest changes..."
    if [ "$(id -u)" -eq 0 ]; then
        chown -R "$TARGET_USER" "$REPO_DIR"
    fi
    cd "$REPO_DIR"
    sudo -u "$TARGET_USER" git pull
fi

cd "$REPO_DIR"

# 5. Install Dependencies (preserving remote machine's JAX)
echo "Setting up Python environment..."

# Create virtual environment with system site packages to inherit pre-installed JAX
VENV_DIR="$REPO_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR (with --system-site-packages)..."
    sudo -u "$TARGET_USER" python3 -m venv --system-site-packages "$VENV_DIR"
fi

PYTHON_CMD="$VENV_DIR/bin/python"
PIP_CMD="$VENV_DIR/bin/pip"

# Check JAX version via virtual environment
JAX_VERSION=$(sudo -u "$TARGET_USER" $PYTHON_CMD -c "import jax; print(jax.__version__)" 2>/dev/null || true)
JAXLIB_VERSION=$(sudo -u "$TARGET_USER" $PYTHON_CMD -c "import jaxlib; print(jaxlib.__version__)" 2>/dev/null || true)

if [ -n "$JAX_VERSION" ]; then
    echo "Found pre-installed JAX version: $JAX_VERSION"
    echo "Found pre-installed jaxlib version: $JAXLIB_VERSION"
    
    # Generate constraints.txt dynamically to pin JAX version
    CONSTRAINTS_FILE="constraints.txt"
    echo "jax==$JAX_VERSION" > "$CONSTRAINTS_FILE"
    if [ -n "$JAXLIB_VERSION" ]; then
        echo "jaxlib==$JAXLIB_VERSION" >> "$CONSTRAINTS_FILE"
    fi
    if [ "$(id -u)" -eq 0 ]; then
        chown "$TARGET_USER" "$CONSTRAINTS_FILE"
    fi
    
    echo "Installing requirements from requirements.txt (excluding/pinning JAX)..."
    sudo -u "$TARGET_USER" $PIP_CMD install -c "$CONSTRAINTS_FILE" -r requirements.txt
    rm -f "$CONSTRAINTS_FILE"
else
    echo "JAX not found in pre-installed environment. Installing default requirements..."
    sudo -u "$TARGET_USER" $PIP_CMD install -r requirements.txt
fi

# 6. Download Dataset (FineWeb-Edu & HellaSwag)
echo "Checking datasets..."

# FineWeb-Edu Check
FINEWEB_DIR="edu_fineweb10B"
HAS_FINEWEB=false
if [ -d "$FINEWEB_DIR" ]; then
    # Check for at least 1 train and 1 val shard file
    if ls "$FINEWEB_DIR"/edufineweb_train_*.npy 1>/dev/null 2>&1 && ls "$FINEWEB_DIR"/edufineweb_val_*.npy 1>/dev/null 2>&1; then
        HAS_FINEWEB=true
    fi
fi

if [ "$HAS_FINEWEB" = true ]; then
    echo "FineWeb-Edu dataset is already present."
else
    echo "FineWeb-Edu dataset not found or incomplete. Starting download..."
    # Note: By default, we download shards for the first 100 steps (2 shards).
    sudo -u "$TARGET_USER" $PYTHON_CMD fineweb.py
fi

# HellaSwag Check
HELLASWAG_DIR="hellaswag"
HAS_HELLASWAG=false
if [ -d "$HELLASWAG_DIR" ]; then
    if [ -f "$HELLASWAG_DIR/hellaswag_train.jsonl" ] && [ -f "$HELLASWAG_DIR/hellaswag_val.jsonl" ]; then
        HAS_HELLASWAG=true
    fi
fi

if [ "$HAS_HELLASWAG" = true ]; then
    echo "HellaSwag dataset is already present."
else
    echo "HellaSwag dataset not found. Downloading..."
    sudo -u "$TARGET_USER" $PYTHON_CMD -c "import hellaswag; hellaswag.download('val'); hellaswag.download('train'); hellaswag.download('test')"
fi

echo "=== Setup Completed Successfully! ==="
