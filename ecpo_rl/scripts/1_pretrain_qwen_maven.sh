#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE_CONFIG="${ROOT_DIR}/ecpo_rl/config.yaml"
eval "$(python "${ROOT_DIR}/ecpo_rl/scripts/resolve_mode_env.py" --config "${MODE_CONFIG}")"

export WANDB_PROJECT="${ECPO_WANDB_PROJECT}"
export WANDB_MODE="online"              # offline/online
# 可选：团队与标签
# export WANDB_ENTITY="your_team"
export WANDB_TAGS="${ECPO_WANDB_SFT_TAGS}"
# 可选：日志目录
# export WANDB_DIR="$(pwd)/wandb"
# 若之前离线过，这条会切回在线；失败不致命
wandb online >/dev/null 2>&1 || true

if [[ "${ECPO_PRETRAIN_CONFIG}" = /* ]]; then
  CONFIG_PATH="${ECPO_PRETRAIN_CONFIG}"
else
  CONFIG_PATH="${ROOT_DIR}/${ECPO_PRETRAIN_CONFIG}"
fi

if [[ "${ECPO_EXPORT_CONFIG}" = /* ]]; then
  EXPORT_CONFIG="${ECPO_EXPORT_CONFIG}"
else
  EXPORT_CONFIG="${ROOT_DIR}/${ECPO_EXPORT_CONFIG}"
fi

echo "$CONFIG_PATH"; test -f "$CONFIG_PATH"


cd "${ROOT_DIR}"

# if [ ! -f "data/processed/maven_sft_train.jsonl" ]; then
  # echo "[INFO] 数据未找到，自动生成演示样本。"
  # python ecpo_rl/scripts/0_convert_maven_to_event_traj.py
# fi

if [ ! -f "requirements.txt" ]; then
  echo "[ERROR] requirements.txt 不存在"
  exit 1
fi

# python -m pip install --upgrade pip >/dev/null
# python -m pip install -r requirements.txt

echo "[INFO] 当前模式：${ECPO_MODE}，启动 SFT 训练"
llamafactory-cli train "${CONFIG_PATH}"

if [ -f "${EXPORT_CONFIG}" ]; then
  echo "[INFO] 合并 LoRA 权重生成全量模型"
  llamafactory-cli export "${EXPORT_CONFIG}"
else
  echo "[WARN] 未找到导出配置 ${EXPORT_CONFIG}，跳过合并步骤"
fi
