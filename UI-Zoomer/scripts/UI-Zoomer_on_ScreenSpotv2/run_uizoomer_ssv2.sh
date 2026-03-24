#!/bin/bash

echo "🧹 [1/4] Cleaning up GPU memory..."
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
echo "✅ Cleanup complete"

MODEL_PATH="/root/autodl-tmp/models/UI-Venus-7B"

IMG_PATH="/root/autodl-tmp/benchmark/ScreenSpot-v2/screenspotv2_image"
TEST_FILES="/root/autodl-tmp/benchmark/ScreenSpot-v2/screenspot_desktop_v2.json,/root/autodl-tmp/benchmark/ScreenSpot-v2/screenspot_mobile_v2.json,/root/autodl-tmp/benchmark/ScreenSpot-v2/screenspot_web_v2.json"

LOG_DIR="/root/autodl-tmp/logs"
mkdir -p $LOG_DIR

SCRIPT_NAME="UI-Zoomer/uizoomer.py"
SAMPLES=8
THRESHOLD=0.6
SIGMA=4.5
EXP_TAG="ssv2_gaussian_sigma_${SIGMA}"

echo "🚀 Starting Experiment: ${EXP_TAG}"

for i in {0..3}; do
  JSON_PATH="${LOG_DIR}/${EXP_TAG}_s${SAMPLES}_part${i}.json"
  TEXT_LOG="${LOG_DIR}/run_log_${EXP_TAG}_s${SAMPLES}_part${i}.txt"

  echo "▶️ GPU $i -> Chunk $i -> Log: $TEXT_LOG"

  CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
    --model_name_or_path "$MODEL_PATH" \
    --screenspot_imgs "$IMG_PATH" \
    --screenspot_test "$TEST_FILES" \
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
