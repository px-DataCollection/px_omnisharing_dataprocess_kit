set -euo pipefail
echo "Please run: \$source env.sh first"
# Ensure envsubst exists
if ! command -v envsubst >/dev/null 2>&1; then
  echo "ERROR: 'envsubst' not found. Install it (Ubuntu): sudo apt-get update && sudo apt-get install -y gettext-base"
  exit 1
fi

SCRIPT_DIR="$(pwd)"

# 1) Install the wheel
pip install --force-reinstall px_retargeting-1.1.0-cp310-cp310-linux_x86_64.whl

# 2) Install third party dependencies
echo "==> Inflating the packages"
cd third_party
unzip -q manotorch.zip
unzip -q pytorch_kinematics.zip

echo "==> Installing third_party/manotorch"
cd manotorch
python -m pip install .

echo "==> Installing third_party/pytorch_kinematics"
cd ../pytorch_kinematics
python -m pip install .
cd ../../

# 3) Generate config JSONs
echo "==> Generating retarget.json files with envsubst"
envsubst < config/retarget/exrh_to_dh13rh_pre.json \
  > config/retarget/exrh_to_dh13rh.json

envsubst < config/retarget/exlh_to_dh13lh_pre.json \
  > config/retarget/exlh_to_dh13lh.json

echo "Installation complete"

