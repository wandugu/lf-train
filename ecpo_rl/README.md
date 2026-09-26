## Overview

This directory contains the **ECPO** (Evidence-Coupled Policy Optimization) workflow built on top of LlamaFactory. ECPO trains a policy to produce an auditable joint output: a ranked candidate list and position-aligned evidence certificates. The goal is not only to rank candidates, but also to ensure that the cited evidence is valid, traceable, and sufficient to recover the ranking decision.

The policy output is strict JSON with three required top-level fields:

- `window_id`: the ranking window identifier.
- `topk`: window-local candidate IDs such as `C001` and `C002`.
- `certificates`: evidence bundles aligned by rank position with `topk`.

Each certificate contains one object per skeleton step. Matched steps cite `doc_id/span` evidence, while unmatched steps must use `matched: false`, `event_id: null`, and an empty evidence list. The default skeleton stages are `PREP -> PROBE -> EXECUTE -> OUTCOME`.

ECPO uses three reward components:

- Listwise ranking utility from the learned trajectory reward `R_theta`.
- Certificate validity reward `r_cert`, based on deterministic schema, span, and traceability checks.
- Evidence-cycle reward `r_cycle`, where a deterministic evidence-only verifier reconstructs candidates from claim-stripped evidence bundles.

Typical commands:

```bash
python ecpo_rl/scripts/0_convert_maven_to_event_traj.py
python ecpo_rl/scripts/0_convert_rams_to_event_traj.py
bash ecpo_rl/scripts/2_train_reward_maxent.sh
bash ecpo_rl/scripts/3_train_policy_rl.sh
python ecpo_rl/evaluate_ecpo.py
```

The main PPO/GRPO reward callback is `src/llamafactory/plugins/reward_callbacks/ecpo.py`, configured with `reward_callback: ecpo`.ECPO 训练与评估说明

