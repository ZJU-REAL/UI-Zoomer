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

# ================= UI-Vision Data Parsing =================

def _get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def extract_bbox_xyxy_from_sample(sample):

    if not isinstance(sample, dict):
        return None

    for key in ["bbox", "box", "bbox_xyxy", "gt_box", "target_box"]:
        v = sample.get(key, None)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                return [float(x) for x in v]
            except:
                pass

    keys_str = all(k in sample for k in ["0", "1", "2", "3"])
    keys_int = all(k in sample for k in [0, 1, 2, 3])
    if keys_str:
        try:
            return [float(sample["0"]), float(sample["1"]), float(sample["2"]), float(sample["3"])]
        except:
            return None
    if keys_int:
        try:
            return [float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3])]
        except:
            return None

    return None

def clamp_xyxy(box_xyxy, img_w, img_h):
    if box_xyxy is None or len(box_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(x) for x in box_xyxy]
    
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    return [x1, y1, x2, y2]

def load_uivision_files(test_files):
    files = [p.strip() for p in test_files.split(",") if p.strip()]
    bundles = []
    for fp in files:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"UI-Vision json not found: {fp}")
        with open(fp, "r") as f:
            data = json.load(f)

        # split name from file name
        split = os.path.splitext(os.path.basename(fp))[0]  # e.g. element_grounding_basic
        bundles.append((split, data))
    return bundles

def normalize_sample(sample, split_name):
    
    if not isinstance(sample, dict):
        return None

    img_relpath = _get_first(sample, ["image_path", "img_filename", "img_path", "image", "filename"])
    instruction = _get_first(sample, ["prompt_to_evaluate", "instruction", "query", "prompt"])

    element_type = _get_first(sample, ["element_type", "type", "data_type"], default="unknown")
    platform = _get_first(sample, ["platform"], default="unknown")
    category = _get_first(sample, ["category"], default="unknown")

    bbox_xyxy = extract_bbox_xyxy_from_sample(sample)

    if not img_relpath or not instruction or bbox_xyxy is None:
        return None

    return {
        "split": split_name,
        "img_relpath": img_relpath,
        "instruction": instruction,
        "bbox_xyxy": bbox_xyxy,
        "element_type": element_type,
        "platform": platform,
        "category": category,
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

def eval_point_in_gt_xyxy(pred_point_norm, gt_xyxy, img_w, img_h):
    if pred_point_norm is None or gt_xyxy is None:
        return "wrong"
    gt = clamp_xyxy(gt_xyxy, img_w, img_h)
    if gt is None:
        return "wrong"
    x1, y1, x2, y2 = gt
    px = pred_point_norm[0] * img_w
    py = pred_point_norm[1] * img_h
    return "correct" if (x1 <= px <= x2) and (y1 <= py <= y2) else "wrong"

# ================= Main =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)

    # UI-Vision
    parser.add_argument("--uivision_imgs", type=str, required=True)   # e.g. /root/autodl-tmp/benchmark/ui-vision/images
    parser.add_argument("--uivision_test", type=str, required=True)   # comma-separated json files
    parser.add_argument("--log_path", type=str, required=True)

    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)

    # keep consistent (unused)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--gating_threshold", type=float, default=1.5)
    parser.add_argument("--sigma_scale", type=float, default=3.5)

    return parser.parse_args()

def main(args):
    bundles = load_uivision_files(args.uivision_test)

    tasks_all = []
    for split_name, samples in bundles:

        if isinstance(samples, dict):

            samples_list = _get_first(samples, ["data", "annotations", "samples", "items"], default=None)
            if isinstance(samples_list, list):
                samples = samples_list

        if not isinstance(samples, list):
            logging.warning(f"[{split_name}] Unsupported json root type: {type(samples)}")
            continue

        for s in samples:
            ns = normalize_sample(s, split_name)
            if ns is None:
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
    print("Starting UI-Vision Baseline (vLLM) Inference...")

    for sample in tqdm(tasks_to_run):
        img_path = os.path.join(args.uivision_imgs, sample["img_relpath"])

        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
        except:
            results.append({
                "split": sample["split"],
                "img_relpath": sample["img_relpath"],
                "img_path": img_path,
                "instruction": sample["instruction"],
                "bbox_xyxy": sample["bbox_xyxy"],
                "platform": sample.get("platform", "unknown"),
                "category": sample.get("category", "unknown"),
                "element_type": sample.get("element_type", "unknown"),
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
            correctness = eval_point_in_gt_xyxy(pred_point, sample["bbox_xyxy"], img_w, img_h)

        results.append({
            "split": sample["split"],
            "img_relpath": sample["img_relpath"],
            "img_path": img_path,
            "instruction": sample["instruction"],
            "bbox_xyxy": sample["bbox_xyxy"],
            "platform": sample.get("platform", "unknown"),
            "category": sample.get("category", "unknown"),
            "element_type": sample.get("element_type", "unknown"),
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
