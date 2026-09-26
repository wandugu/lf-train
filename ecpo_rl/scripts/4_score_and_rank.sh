#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE_CONFIG="${ROOT_DIR}/ecpo_rl/config.yaml"
eval "$(python "${ROOT_DIR}/ecpo_rl/scripts/resolve_mode_env.py" --config "${MODE_CONFIG}")"

OUTPUT_DIR="${ECPO_POLICY_OUTPUT_DIR}"
REWARD_CKPT="${ECPO_REWARD_CKPT}"
TRAJ_PATH="${ROOT_DIR}/${ECPO_PROCESSED_DIR}/${ECPO_TRAJ_FILE}"
POLICY_LOGPROB="${OUTPUT_DIR}/policy_logprobs.json"

cd "${ROOT_DIR}"

if [ ! -f "${REWARD_CKPT}" ]; then
  echo "[ERROR] 奖励模型缺失，请先运行 2_train_reward_maxent.sh"
  exit 1
fi

if [ ! -f "${POLICY_LOGPROB}" ]; then
  echo "[WARN] 未找到策略 logprob，启用启发式得分"
fi

python ecpo_rl/policy/score_policy.py \
  --traj-path "${TRAJ_PATH}" \
  --reward-ckpt "${REWARD_CKPT}" \
  --policy-logprobs "${POLICY_LOGPROB}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
