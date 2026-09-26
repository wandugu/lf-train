#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE_CONFIG="${ROOT_DIR}/ecpo_rl/config.yaml"
eval "$(python "${ROOT_DIR}/ecpo_rl/scripts/resolve_mode_env.py" --config "${MODE_CONFIG}")"

TRAJ_PATH="${ROOT_DIR}/${ECPO_PROCESSED_DIR}/${ECPO_TRAJ_FILE}"
PAIR_PATH="${ROOT_DIR}/${ECPO_PROCESSED_DIR}/${ECPO_PAIRS_FILE}"
OUTPUT_DIR="${ECPO_REWARD_OUTPUT_DIR}"
OUTPUT_PATH="${ECPO_REWARD_CKPT}"

cd "${ROOT_DIR}"

if [ ! -f "${TRAJ_PATH}" ]; then
  echo "[INFO] 轨迹文件缺失，先生成演示数据"
fi

echo "[INFO] 当前模式：${ECPO_MODE}"

mkdir -p "${OUTPUT_DIR}"

python ecpo_rl/irl/maxent_irl.py \
  --traj-path "${TRAJ_PATH}" \
  --pair-path "${PAIR_PATH}" \
  --output "${OUTPUT_PATH}" \
  --epochs 150 \
  --lr 0.05
