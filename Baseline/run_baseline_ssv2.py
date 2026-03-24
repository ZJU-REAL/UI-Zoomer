import os
import re
import json
import math
import argparse
import logging
from tqdm import tqdm
from PIL import Image
import torch

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

# ================= SSV2 Data Parsing =================

def extract_bbox_xywh_from_sample(sample):
    if isinstance(sample, dict):
        if "bbox" in sample and isinstance(sample["bbox"], (list, tuple)) and len(sample["bbox"]) == 4:
            return [float(x) for x in sample["bbox"]]
        if "box" in sample and isinstance(sample["box"], (list, tuple)) and len(sample["box"]) == 4:
            return [float(x) for x in sample["box"]]

        keys_str = all(k in sample for k in ["0", "1", "2", "3"])
        keys_int = all(k in sample for k in [0, 1, 2, 3])
        if keys_str:
            return [float(sample["0"]), float(sample["1"]), float(sample["2"]), float(sample["3"])]
        if keys_int:
            return [float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3])]
    return None

def xywh_to_xyxy(bbox_xywh, img_w, img_h):
    if bbox_xywh is None or len(bbox_xywh) != 4:
        return None
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return [x1, y1, x2, y2]

def load_ssv2_files(test_files):
    files = [p.strip() for p in test_files.split(",") if p.strip()]
    bundles = []
    for fp in files:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"SSV2 json not found: {fp}")
        with open(fp, "r") as f:
            data = json.load(f)
        base = os.path.basename(fp).lower()
        if "desktop" in base:
            split = "desktop"
        elif "mobile" in base:
            split = "mobile"
        elif "web" in base:
            split = "web"
        else:
            split = os.path.splitext(os.path.basename(fp))[0]
        bundles.append((split, data))
    return bundles

def normalize_sample(sample, split_name):
    img_filename = sample.get("img_filename") or sample.get("image") or sample.get("img") or sample.get("filename")
    instruction = sample.get("instruction") or sample.get("instruction_en") or sample.get("query") or sample.get("prompt")
    data_type = sample.get("data_type") or sample.get("type") or "unknown"
    data_source = sample.get("data_source") or sample.get("source") or "unknown"
    bbox_xywh = extract_bbox_xywh_from_sample(sample)
    return {
        "split": split_name,
        "img_filename": img_filename,
        "instruction": instruction,
        "data_type": data_type,
        "data_source": data_source,
        "bbox_xywh": bbox_xywh,
    }

# ================= vLLM Baseline Model =================

class UI_Venus_Ground_vLLM_Baseline:
    def load_model(self, model_name_or_path):
        print(f"[vLLM-Baseline] Loading model from {model_name_or_path}...")
        mm_processor_kwargs = {
            "min_pixels": 10000,
            "max_pixels": 5000000,
        }
        self.llm = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 1},
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            mm_processor_kwargs=mm_processor_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    def inference_one(self, instruction, image):

        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=128,
            stop=["<|endoftext|>", "<|im_end|>"],
            logprobs=0
        )

        prompt_origin = "Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2]."
        full_instruction = prompt_origin.format(instruction)

        prompt_text = (
            f"<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{full_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = {"prompt": prompt_text, "multi_modal_data": {"image": image}}

        outputs = self.llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        w, h = image.size
        box = self._parse_box_norm(text, w, h)
        return {"text": text, "box_norm": box}

    def _parse_box_norm(self, text, w, h):
        try:
            match = re.search(r"\[(.*?)\]", text)
            if not match:
                return None
            content = match.group(1).replace(",", " ")
            coords = [float(x) for x in content.split() if x.strip()]
            if len(coords) != 4:
                return None
            x1, x2 = sorted([coords[0], coords[2]])
            y1, y2 = sorted([coords[1], coords[3]])
            return [
                max(0.0, min(1.0, x1 / w)),
                max(0.0, min(1.0, y1 / h)),
                max(0.0, min(1.0, x2 / w)),
                max(0.0, min(1.0, y2 / h)),
            ]
        except:
            return None

# ================= Eval =================

def eval_point_in_gt_xywh(pred_point_norm, bbox_xywh, img_w, img_h):
    if pred_point_norm is None or bbox_xywh is None:
        return "wrong"
    gt_xyxy = xywh_to_xyxy(bbox_xywh, img_w, img_h)
    if gt_xyxy is None:
        return "wrong"
    x1, y1, x2, y2 = gt_xyxy
    px = pred_point_norm[0] * img_w
    py = pred_point_norm[1] * img_h
    return "correct" if (x1 <= px <= x2) and (y1 <= py <= y2) else "wrong"

# ================= Main =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--screenspot_imgs", type=str, required=True)
    parser.add_argument("--screenspot_test", type=str, required=True)  # comma-separated
    parser.add_argument("--log_path", type=str, required=True)

    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)

    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--gating_threshold", type=float, default=1.5)
    parser.add_argument("--sigma_scale", type=float, default=3.5)

    return parser.parse_args()

def main(args):
    bundles = load_ssv2_files(args.screenspot_test)

    tasks_all = []
    for split_name, samples in bundles:
        for s in samples:
            ns = normalize_sample(s, split_name)
            if not ns["img_filename"] or not ns["instruction"] or not ns["bbox_xywh"]:
                continue
            tasks_all.append(ns)

    # chunk split
    if args.num_chunks > 1:
        total = len(tasks_all)
        chunk_size = math.ceil(total / args.num_chunks)
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, total)
        tasks_to_run = tasks_all[start:end]

        base, ext = os.path.splitext(args.log_path)
        args.log_path = f"{base}_part{args.chunk_id}{ext}"
        print(f"[Chunk {args.chunk_id}/{args.num_chunks}] Total={total}, ThisChunk={len(tasks_to_run)}")
    else:
        tasks_to_run = tasks_all
        print(f"[Single Chunk] Total={len(tasks_to_run)}")

    model = UI_Venus_Ground_vLLM_Baseline()
    model.load_model(args.model_name_or_path)

    results = []
    print("Starting SSV2 Baseline (vLLM) Inference...")

    for i, sample in enumerate(tqdm(tasks_to_run)):
        img_path = os.path.join(args.screenspot_imgs, sample["img_filename"])

        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except:
            results.append({
                "split": sample["split"],
                "data_type": sample["data_type"],
                "data_source": sample.get("data_source", "unknown"),
                "img_filename": sample["img_filename"],
                "img_path": img_path,
                "instruction": sample["instruction"],
                "bbox_xywh": sample["bbox_xywh"],
                "method": "baseline_fail_open",
                "correctness": "wrong",
            })
            continue

        out = model.inference_one(sample["instruction"], img)
        box = out.get("box_norm")

        if box is None:
            correctness = "wrong_format"
            pred_point = None
        else:
            pred_point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            correctness = eval_point_in_gt_xywh(pred_point, sample["bbox_xywh"], img_w, img_h)

        results.append({
            "split": sample["split"],
            "data_type": sample["data_type"],
            "data_source": sample.get("data_source", "unknown"),
            "img_filename": sample["img_filename"],
            "img_path": img_path,
            "instruction": sample["instruction"],
            "bbox_xywh": sample["bbox_xywh"],
            "model_text": out.get("text", ""),
            "method": "baseline_vllm_single",
            "correctness": correctness,
        })

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Finished. Saved to {args.log_path}")

if __name__ == "__main__":
    main(parse_args())
