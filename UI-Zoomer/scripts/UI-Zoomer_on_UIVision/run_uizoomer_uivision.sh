#!/bin/bash

echo "🧹 [1/4] Cleaning up GPU memory..."
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
echo "✅ Cleanup complete"

MODEL_PATH="/root/autodl-tmp/models/UI-Venus-7B"

# UI-Vision
IMG_PATH="/root/autodl-tmp/benchmark/ui-vision/images"
TEST_FILES="/root/autodl-tmp/benchmark/ui-vision/annotations/element_grounding/element_grounding_basic.json,\
            /root/autodl-tmp/benchmark/ui-vision/annotations/element_grounding/element_grounding_functional.json,\
            /root/autodl-tmp/benchmark/ui-vision/annotations/element_grounding/element_grounding_spatial.json,\
            /root/autodl-tmp/benchmark/ui-vision/annotations/layout_grounding/layout_grounding.json"

LOG_DIR="/root/autodl-tmp/logs"
mkdir -p $LOG_DIR

SCRIPT_NAME="UI-Zoomer/uizoomer.py"
SAMPLES=8
THRESHOLD=1.5
SIGMA=1.5

EXP_TAG="uivision_prozoom_thr${THRESHOLD}_sigma${SIGMA}"

echo "🚀 Starting Experiment: ${EXP_TAG}"

for i in {0..3}; do
  JSON_PATH="${LOG_DIR}/${EXP_TAG}_s${SAMPLES}_part${i}.json"
  TEXT_LOG="${LOG_DIR}/run_log_${EXP_TAG}_s${SAMPLES}_part${i}.txt"

  echo "▶️ GPU $i -> Chunk $i -> Log: $TEXT_LOG"

  CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
    --model_name_or_path "$MODEL_PATH" \
    --uivision_imgs "$IMG_PATH" \
    --uivision_test "$TEST_FILES" \
    --log_path "$JSON_PATH" \
    --chunk_id $i \
    --num_chunks 4 \
    --num_samples $SAMPLES \
    --gating_threshold $THRESHOLD \
    --sigma_scale $SIGMA \
    > "$TEXT_LOG" 2>&1 &
done

echo "⏳ [3/4] Tasks started, currently computing..."
wait
echo "✅ Tasks completed."
