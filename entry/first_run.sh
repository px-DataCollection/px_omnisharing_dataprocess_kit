#!/usr/bin/env bash
#
# PaXini EID Pipeline
# Entering script for Docker image
# Last Modified: 2026-01-27

set -euo pipefail

cat << 'EOF'

PaXini EID Pipeline. ver 0.1.3
Launching and (re-)loading the image
Previous container named 'px_pipeline' will be removed

EOF

read -r -p "Press Enter to confirm..." _

tput civis 2>/dev/null || true
total=3
for ((i=total;i>0;i--)); do
  done=$((total - i))
  bar=$(printf '%0.s#' $(seq 1 $done))
  printf "\r[Load px_pipeline in] %d  |%-5s|" "$i" "$bar"
  sleep 1
done
printf "\r\033[K"
tput cnorm 2>/dev/null || true
echo -e "\n==> Attempting to remove 'px_pipeline'"
docker rm -f px_pipeline >/dev/null 2>&1

IMAGE="${IMAGE:-px_pipeline:0.1.3}"    
NAME="${NAME:-px_pipeline}"      
GPUS="${GPUS:-all}"           
WORKDIR="${1:-}"

if [ -z "${WORKDIR}" ]; then
  echo "Usage: $0 /absolute/path/to/workdir [optional command ...]" >&2
  exit 1
fi

# Resolve to absolute path
WORKDIR="$(cd "${WORKDIR}" && pwd)"

shift || true

docker run -it \
  --gpus "${GPUS:-all}" \
  --network=host \
  -v "${WORKDIR:?set WORKDIR to absolute path}":/ws \
  -w /home/app \
  --name "${NAME}" \
  "${IMAGE}" bash -lc '
    set -euo pipefail

    echo "==> Initialization finished. Dropping you into an interactive shell..."
    exec bash -i
  '

