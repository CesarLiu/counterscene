#!/usr/bin/env bash
# Controllability check for the CounterScene counterfactual guidance.
#
# Five arms over one scene set. Two are unguided and differ only in sampling
# seed: their difference is the noise floor, and no control claim is meaningful
# below it. The three guided arms sweep the V3 late-stage multiplier, so the
# intervention should grow with dose while agents outside the controlled set
# stay at the floor.
#
# Set COUNTERSCENE_REPLAY_EGO=1 to hold the ego on its logged trajectory
# instead of letting the world model drive it.
#
# Modified by the CounterScene authors, 2026.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NUSCENES_ROOT:?Set NUSCENES_ROOT to the nuScenes dataset directory}"
: "${COUNTERSCENE_CHECKPOINT_DIR:?Set COUNTERSCENE_CHECKPOINT_DIR}"
: "${COUNTERSCENE_CHECKPOINT_KEY:?Set COUNTERSCENE_CHECKPOINT_KEY}"

OUT="${COUNTERSCENE_RESULTS_DIR:-${ROOT}/results/controllability}"
ARTIFACT="${COUNTERSCENE_SELECTED_VEHICLES:-${ROOT}/data/counterscene_selected_vehicles.json}"
CFG_DIR="${OUT}/configs"
STEPS="${COUNTERSCENE_SIMULATION_STEPS:-50}"
mkdir -p "${CFG_DIR}"

# nuScenes val scenes whose published pair has a moving adversary, ordered by
# the mined encounter distance. The first eight are "tier A" -- d_min < 15 m,
# i.e. the conflict geometry is close enough for the safety margin to move.
EVAL_SCENES="90, 96, 91, 6, 98, 95, 32, 12, 97, 84, 70, 11, 34, 33, 37, 1"

cat > "${CFG_DIR}/guided.json" <<EOF
{"registered_name": "trajdata_nusc_counterscene",
 "trajdata": {"eval_scenes": [${EVAL_SCENES}], "num_scenes_to_evaluate": 16}}
EOF
cat > "${CFG_DIR}/unguided.json" <<EOF
{"registered_name": "trajdata_nusc_counterscene",
 "trajdata": {"eval_scenes": [${EVAL_SCENES}], "num_scenes_to_evaluate": 16},
 "apply_guidance": false}
EOF

REPLAY=()
if [ "${COUNTERSCENE_REPLAY_EGO:-0}" = "1" ]; then REPLAY=(--replay_ego); fi

run_arm () {
  tag="$1"; cfg="$2"; seed="$3"; shift 3
  echo "[$(date +%H:%M:%S)] start ${tag}"
  python "${ROOT}/ccdiff/examples/scene_editor.py" \
    --config_file "${cfg}" \
    --results_root_dir "${OUT}/${tag}" \
    --num_scenes_per_batch 1 \
    --dataset_path "${NUSCENES_ROOT}" \
    --env trajdata \
    --policy_ckpt_dir "${COUNTERSCENE_CHECKPOINT_DIR}" \
    --policy_ckpt_key "${COUNTERSCENE_CHECKPOINT_KEY}" \
    --eval_class CCDiff \
    --editing_source conflict \
    --selected_vehicles_path "${ARTIFACT}" \
    --registered_name trajdata_nusc_counterscene \
    --part_control --controllable_agent 2 \
    --num_simulation_steps "${STEPS}" \
    --n_step_action 5 --save_every_n_frames 5 \
    --seed "${seed}" "${REPLAY[@]}" "$@" > "${OUT}/${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] done ${tag} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then
    echo "  !! ${tag} did not finish -- its rollout covers fewer scenes than the"
    echo "     others, so score it only over the scenes every arm has. Last lines:"
    tail -3 "${OUT}/${tag}.log" | sed 's/^/     /'
  fi
}

# The arms are independent, but do NOT run all five at once on a single card.
# Per-process memory scales with the number of agents in the current scene, so
# five arms fit while the early scenes are small and then OOM part-way through
# the set -- each arm dies at a different scene and the usable intersection
# shrinks silently. Two arms on a 24 GB card is safe; serial is the default.
run_arm baseline     "${CFG_DIR}/unguided.json" 0
run_arm baseline_s1  "${CFG_DIR}/unguided.json" 1
run_arm conservative "${CFG_DIR}/guided.json"   0 --v3_ablation_variant very_conservative
run_arm full         "${CFG_DIR}/guided.json"   0 --v3_ablation_variant full
run_arm aggressive   "${CFG_DIR}/guided.json"   0 --v3_ablation_variant very_aggressive

echo
echo "score each arm against the unguided baseline:"
for tag in baseline_s1 conservative full aggressive; do
  echo "  python ${ROOT}/scripts/score_rollout.py \\"
  echo "    --hdf5 ${OUT}/${tag}/scene_edit_eval/data.hdf5 \\"
  echo "    --reference_hdf5 ${OUT}/baseline/scene_edit_eval/data.hdf5 \\"
  echo "    --selected_vehicles ${ARTIFACT} --output ${OUT}/metrics_${tag}.json"
done
