import copy
import itertools
import torch
import json
import re
import argparse
import os
import math
import numpy as np
from PIL import Image, ImageDraw
import logging
from tqdm import tqdm

# ==================== vLLM Model Class ====================
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class UI_Venus_Ground_vLLM_Gating():
    def load_model(self, model_name_or_path):
        print(f"[vLLM-Gating] Loading model from {model_name_or_path}...")
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

    def inference(self, instruction, image, k=1, temperature=0.9):
        if k > 1:
            sampling_params = SamplingParams(
                n=k, temperature=temperature, max_tokens=128, 
                stop=["<|endoftext|>", "<|im_end|>"], logprobs=1
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0, max_tokens=128, 
                stop=["<|endoftext|>", "<|im_end|>"], logprobs=1
            )

        prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
        full_instruction = prompt_origin.format(instruction)
        
        prompt_text = (
            f"<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{full_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = {"prompt": prompt_text, "multi_modal_data": {"image": image}}
        
        try:
            outputs = self.llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
            candidates = []
            final_w, final_h = image.size 
            
            for output_item in outputs[0].outputs:
                text = output_item.text.strip()
                confidence = 0.0
                if output_item.logprobs:
                    token_logprobs = []
                    for i, step_logprobs_dict in enumerate(output_item.logprobs):
                        if i < len(output_item.token_ids):
                            token_id = output_item.token_ids[i]
                            if token_id in step_logprobs_dict:
                                token_logprobs.append(step_logprobs_dict[token_id].logprob)
                    if token_logprobs:
                        avg_logprob = sum(token_logprobs) / len(token_logprobs)
                        confidence = math.exp(avg_logprob)

                box = self._parse_box(text, final_w, final_h)
                candidates.append({"text": text, "box": box, "confidence": confidence})
            return candidates
        except Exception as e:
            print(f"vLLM Error: {e}")
            return []

    def _parse_box(self, text, w, h):
        try:
            match = re.search(r'\[(.*?)\]', text)
            if match:
                content = match.group(1).replace(',', ' ')
                coords = [float(x) for x in content.split() if x.strip()]
                if len(coords) == 4:
                    x1, x2 = sorted([coords[0], coords[2]])
                    y1, y2 = sorted([coords[1], coords[3]])
                    return [
                        max(0.0, min(1.0, x1 / w)), max(0.0, min(1.0, y1 / h)),
                        max(0.0, min(1.0, x2 / w)), max(0.0, min(1.0, y2 / h))
                    ]
            return None
        except:
            return None

# ================= Helper Functions =================

def smart_resize_target(height, width, min_pixels, max_pixels):
    current_pixels = height * width
    factor = 1.0
    if current_pixels < min_pixels:
        factor = math.sqrt(min_pixels / current_pixels)
    elif current_pixels > max_pixels:
        factor = math.sqrt(max_pixels / current_pixels)
    new_h = int(height * factor)
    new_w = int(width * factor)
    align = 28
    new_h = (new_h + align - 1) // align * align
    new_w = (new_w + align - 1) // align * align
    return new_w, new_h

def calculate_iou(box1, box2):
    if box1 is None or box2 is None: return 0.0
    b1_x1, b1_x2 = min(box1[0], box1[2]), max(box1[0], box1[2])
    b1_y1, b1_y2 = min(box1[1], box1[3]), max(box1[1], box1[3])
    b2_x1, b2_x2 = min(box2[0], box2[2]), max(box2[0], box2[2])
    b2_y1, b2_y2 = min(box2[1], box2[3]), max(box2[1], box2[3])

    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union = area1 + area2 - intersection
    if union <= 1e-6: return 0.0
    return intersection / union

def calculate_spatial_consistency(boxes):
    N = len(boxes)
    if N <= 1: return 1.0
    iou_sum = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            if i == j: continue
            iou_sum += calculate_iou(boxes[i], boxes[j])
            count += 1
    if count == 0: return 0.0
    return iou_sum / count

# Density Filtering + Gaussian Crop 
def get_density_gaussian_crop_box(candidates, orig_w, orig_h, sigma_scale=3.5, min_crop_size=512):
    if not candidates: return None
    
    valid_data = []
    for c in candidates:
        if c['box']:
            bx1, by1, bx2, by2 = c['box']
            bx1, bx2 = bx1 * orig_w, bx2 * orig_w
            by1, by2 = by1 * orig_h, by2 * orig_h
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            w, h = bx2 - bx1, by2 - by1
            valid_data.append({"center": [cx, cy], "size": [w, h], "box": [bx1, by1, bx2, by2]})
            
    if not valid_data: return None
    
    # Density Filtering - Top 75%
    centers_all = np.array([d["center"] for d in valid_data])
    median_center = np.median(centers_all, axis=0)
    distances = np.linalg.norm(centers_all - median_center, axis=1)
    
    keep_k = max(1, int(len(valid_data) * 0.75))
    sorted_indices = np.argsort(distances)
    keep_indices = sorted_indices[:keep_k]
    
    filtered_centers = centers_all[keep_indices]
    filtered_sizes = np.array([valid_data[i]["size"] for i in keep_indices])
    
    # Gaussian Statistics
    # A. Inter-group Variance 
    var_inter = np.var(filtered_centers, axis=0) 
    
    # B. Intra-group Variance 
    sigmas_intra = filtered_sizes / 4.0
    vars_intra = np.square(sigmas_intra)
    avg_var_intra = np.mean(vars_intra, axis=0) 
    
    # C. Total Variance & Sigma
    total_var = var_inter + avg_var_intra
    sigma_total = np.sqrt(total_var) 
    
    # D. Global Center
    mu_global = np.mean(filtered_centers, axis=0) 
    
    # Crop Box Calculation
    radius_w = sigma_scale * sigma_total[0]
    radius_h = sigma_scale * sigma_total[1]
    
    crop_w = radius_w * 2
    crop_h = radius_h * 2
    
    target_w = max(crop_w, min_crop_size)
    target_h = max(crop_h, min_crop_size)
    
    # Crop Box Normalization
    final_size = max(target_w, target_h)
    half_s = final_size / 2.0
    
    final_x1 = mu_global[0] - half_s
    final_x2 = mu_global[0] + half_s
    final_y1 = mu_global[1] - half_s
    final_y2 = mu_global[1] + half_s


    if final_x1 < 0: final_x2 -= final_x1; final_x1 = 0
    if final_y1 < 0: final_y2 -= final_y1; final_y1 = 0
    if final_x2 > orig_w: final_x1 -= (final_x2 - orig_w); final_x2 = orig_w
    if final_y2 > orig_h: final_y1 -= (final_y2 - orig_h); final_y2 = orig_h
    
    final_x1 = max(0, final_x1); final_y1 = max(0, final_y1)
    final_x2 = min(orig_w, final_x2); final_y2 = min(orig_h, final_y2)

    return [
        final_x1/orig_w, final_y1/orig_h, 
        final_x2/orig_w, final_y2/orig_h
    ]



def plot_debug_image(img_path, save_path, gt_bbox_px, candidate_boxes_norm, crop_box_norm, pred_point_norm, score_info):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        if crop_box_norm:
            x1, x2 = sorted([crop_box_norm[0], crop_box_norm[2]])
            y1, y2 = sorted([crop_box_norm[1], crop_box_norm[3]])
            rect = [x1*w, y1*h, x2*w, y2*h]
            draw.rectangle(rect, outline="red", width=4)

        if candidate_boxes_norm:
            for box in candidate_boxes_norm:
                if box:
                    x1, x2 = sorted([box[0], box[2]])
                    y1, y2 = sorted([box[1], box[3]])
                    rect = [x1*w, y1*h, x2*w, y2*h]
                    draw.rectangle(rect, outline="blue", width=2)

        if gt_bbox_px:
            gx1, gx2 = sorted([gt_bbox_px[0], gt_bbox_px[2]])
            gy1, gy2 = sorted([gt_bbox_px[1], gt_bbox_px[3]])
            draw.rectangle([gx1, gy1, gx2, gy2], outline="green", width=3)

        if pred_point_norm:
            px, py = pred_point_norm[0]*w, pred_point_norm[1]*h
            r = 8
            draw.ellipse((px-r, py-r, px+r, py+r), fill="yellow", outline="black")
            
        if score_info:
            text = f"S={score_info['S']:.2f}"
            draw.text((10, 10), text, fill="red")

        img.save(save_path)
    except Exception as e:
        print(f"[Plot Error] {e}")

# ================= Core Processing Logic =================

def process_single_image(model, instruction, img_path, num_samples=8, gating_threshold=1.5, sigma_scale=3.5):
    debug_info = {"candidate_boxes": [], "crop_box": None, "score_info": None}
    
    img_name = os.path.basename(img_path)
    
    try:
        raw_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_img.size
    except:
        return {"result": "error", "point": None, "method": "fail", "debug_info": debug_info}

    # Step 1: Global Sampling
    target_w, target_h = smart_resize_target(orig_h, orig_w, 2000000, 4800000)
    img_resized = raw_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
    
    candidates = model.inference(instruction, img_resized, k=num_samples, temperature=0.9)
    valid_candidates = [c for c in candidates if c['box'] is not None]
    debug_info["candidate_boxes"] = [c['box'] for c in valid_candidates]

    if not valid_candidates:
        return {"result": "no_box", "point": [0.5, 0.5], "method": "fallback_center", "debug_info": debug_info}

    # Debug Log
    log_msg = [f"\n🔍 [DEBUG: {img_name}]"]
    log_msg.append(f"  Candidates ({len(valid_candidates)}):")
    for idx, c in enumerate(valid_candidates):
        log_msg.append(f"    [{idx}] Box: {[f'{x:.2f}' for x in c['box']]}, Conf: {c['confidence']:.4f}")

    # Step 2: Scoring
    boxes = [c['box'] for c in valid_candidates]
    c_spatial = calculate_spatial_consistency(boxes)
    avg_conf = sum(c['confidence'] for c in valid_candidates) / len(valid_candidates)
    S = c_spatial + avg_conf

    
    debug_info["score_info"] = {"C_spatial": c_spatial, "avg_conf": avg_conf, "S": S}
    log_msg.append(f"  📊 Scores: S={S:.2f}")

    # Step 3: Branching
    if S > gating_threshold:
        log_msg.append(f"  ✅ Decision: PASS")
        tqdm.write("\n".join(log_msg))
        
        # Vote Logic
        votes = [0] * len(valid_candidates)
        for i in range(len(valid_candidates)):
            for j in range(len(valid_candidates)):
                if i == j: continue
                if calculate_iou(valid_candidates[i]['box'], valid_candidates[j]['box']) > 0.5:
                    votes[i] += 1
        max_votes = max(votes) if votes else 0
        if max_votes > 0:
            best_indices = [i for i, v in enumerate(votes) if v == max_votes]
            best_idx = max(best_indices, key=lambda i: valid_candidates[i]['confidence'])
            method_str = f"gating_pass_vote (S={S:.2f})"
        else:
            best_idx = max(range(len(valid_candidates)), key=lambda i: valid_candidates[i]['confidence'])
            method_str = f"gating_pass_conf (S={S:.2f})"
        
        final_box = valid_candidates[best_idx]['box']
        center = [(final_box[0]+final_box[2])/2, (final_box[1]+final_box[3])/2]
        return {"result": "success", "point": center, "method": method_str, "debug_info": debug_info}
        
    else:
        log_msg.append(f"  ✂️ Decision: DENSITY GAUSSIAN CROP (Sigma={sigma_scale})")
        
        crop_norm = get_density_gaussian_crop_box(
            valid_candidates, orig_w, orig_h, 
            sigma_scale=sigma_scale, 
            min_crop_size=512
        )
        debug_info["crop_box"] = crop_norm
        
        c_x1, c_y1 = int(crop_norm[0]*orig_w), int(crop_norm[1]*orig_h)
        c_x2, c_y2 = int(crop_norm[2]*orig_w), int(crop_norm[3]*orig_h)
        
        log_msg.append(f"    Gaussian Crop: {c_x2-c_x1}x{c_y2-c_y1}")
        tqdm.write("\n".join(log_msg))
        
        # Crop & Refine
        crop_img = raw_img.crop((c_x1, c_y1, c_x2, c_y2))
        cw, ch = crop_img.size
        target_cw, target_ch = smart_resize_target(ch, cw, 1000000, 4000000)
        crop_img_resized = crop_img.resize((target_cw, target_ch), Image.Resampling.BICUBIC)
        
        final_candidates = model.inference(instruction, crop_img_resized, k=1, temperature=0.0)
        
        if final_candidates and final_candidates[0]['box'] is not None:
            lb = final_candidates[0]['box']
            lx1, ly1 = lb[0]*(c_x2-c_x1), lb[1]*(c_y2-c_y1)
            lx2, ly2 = lb[2]*(c_x2-c_x1), lb[3]*(c_y2-c_y1)
            gx1, gy1 = (lx1+c_x1)/orig_w, (ly1+c_y1)/orig_h
            gx2, gy2 = (lx2+c_x1)/orig_w, (ly2+c_y1)/orig_h
            center = [(gx1+gx2)/2, (gy1+gy2)/2]
            return {"result": "success", "point": center, "method": f"gaussian_crop_s{sigma_scale}_refine (S={S:.2f})", "debug_info": debug_info}
        else:
            best_c = max(valid_candidates, key=lambda x: x['confidence'])
            b = best_c['box']
            return {"result": "success", "point": [(b[0]+b[2])/2, (b[1]+b[3])/2], "method": f"crop_fail_fallback (S={S:.2f})", "debug_info": debug_info}

# ================= Main =================

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--gating_threshold', type=float, default=1.5)
    parser.add_argument('--sigma_scale', type=float, default=3.5) # <--- 新增参数
    args = parser.parse_args()
    return args

def eval_sample_positive_gt(sample, pred_point, img_size):
    bbox = sample["bbox"]
    px = pred_point[0] * img_size[0]
    py = pred_point[1] * img_size[1]
    if (bbox[0] <= px <= bbox[2]) and (bbox[1] <= py <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def main(args):
    vis_dir = os.path.join(os.path.dirname(args.log_path), f"vis_gating_samples{args.num_samples}")
    os.makedirs(vis_dir, exist_ok=True)

    if args.task == "all":
        task_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.screenspot_test) if f.endswith(".json")]
    else:
        task_filenames = args.task.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        path = os.path.join(args.screenspot_test, task_filename + ".json")
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            task_data = json.load(f)
        for task_instance in task_data:
            ti = copy.deepcopy(task_instance)
            ti["task_filename"] = task_filename
            instr = ti.get("instruction") or ti.get("instruction_en")
            if instr and ti.get("gt_type", "positive") == "positive":
                ti["prompt_to_evaluate"] = instr
                tasks_to_run.append(ti)

    if args.num_chunks > 1:
        total = len(tasks_to_run)
        chunk_size = math.ceil(total / args.num_chunks)
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, total)
        tasks_to_run = tasks_to_run[start:end]
        
        base, ext = os.path.splitext(args.log_path)
        args.log_path = f"{base}_part{args.chunk_id}{ext}"
        print(f"[Chunk {args.chunk_id}/{args.num_chunks}] Count: {len(tasks_to_run)}")

    model = UI_Venus_Ground_vLLM_Gating()
    model.load_model(model_name_or_path=args.model_name_or_path)

    results = []
    print(f"Starting Inference (Samples={args.num_samples}, Sigma={args.sigma_scale})...")
    
    for i, sample in enumerate(tqdm(tasks_to_run)):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        
        res = process_single_image(
            model, 
            sample["prompt_to_evaluate"], 
            img_path, 
            num_samples=args.num_samples,
            gating_threshold=args.gating_threshold,
            sigma_scale=args.sigma_scale
        )
        
        try:
            tmp_img = Image.open(img_path)
            img_size = tmp_img.size
        except:
            img_size = (1920, 1080)

        pred_px = None
        if res["point"]:
            pred_px = [res["point"][0] * img_size[0], res["point"][1] * img_size[1]]
            correctness = eval_sample_positive_gt(sample, res["point"], img_size)
        else:
            correctness = "wrong_format"

        debug_info = res.get("debug_info", {})
        save_name = f"{sample['task_filename']}_{i}_{correctness}.jpg"
        save_path = os.path.join(vis_dir, save_name)
        
        plot_debug_image(
            img_path=img_path,
            save_path=save_path,
            gt_bbox_px=sample["bbox"],
            candidate_boxes_norm=debug_info.get("candidate_boxes"),
            crop_box_norm=debug_info.get("crop_box"),
            pred_point_norm=res["point"],
            score_info=debug_info.get("score_info")
        )

        sample_result = {
            "img_path": img_path,
            "method": res["method"],
            "score": debug_info.get("score_info"),
            "bbox": sample["bbox"],
            "correctness": correctness
        }
        results.append(sample_result)
        
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Finished. Saved to {args.log_path}")

if __name__ == "__main__":
    main(parse_args())