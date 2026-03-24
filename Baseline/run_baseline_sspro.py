import copy
import itertools
import torch
import json
import re
import argparse
import os
import math
from PIL import Image
import logging
from tqdm import tqdm

# ==================== vLLM Backend (Client-Side Resize Fix) ====================
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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

class UI_Venus_Ground_vLLM():
    def load_model(self, model_name_or_path):
        print(f"[vLLM] Loading model from {model_name_or_path}...")
        
        mm_processor_kwargs = {
            "min_pixels": 10000,     
            "max_pixels": 10000000,    
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

    def inference(self, instruction, image_path, img_size):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=128, stop=["<|endoftext|>", "<|im_end|>"])

        prompt_origin = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
        full_instruction = prompt_origin.format(instruction)
        
        prompt_text = (
            f"<|im_start|>user\n"
            f"<|vision_start|><|image_pad|><|vision_end|>{full_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size
            
            target_w, target_h = smart_resize_target(orig_h, orig_w, 2000000, 4800000)
            
            if target_w != orig_w or target_h != orig_h:
                image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            
            final_w, final_h = image.size
            
        except:
            return {"point": None, "raw_response": "Image Load Error"}

        inputs = {"prompt": prompt_text, "multi_modal_data": {"image": image}}
        
        try:
            outputs = self.llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
            text = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"vLLM Error: {e}")
            text = ""


        box = self._parse_box(text, final_w, final_h)
        center_pt = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        
        result_type = "positive" if (box != [0.0, 0.0, 0.0, 0.0]) else "negative"
        
        return {"point": center_pt, "raw_response": text, "result": result_type}

    def _parse_box(self, text, w, h):
        try:
            match = re.search(r'\[(.*?)\]', text)
            if match:
                content = match.group(1).replace(',', ' ')
                coords = [float(x) for x in content.split() if x.strip()]
                if len(coords) == 4:

                    x1 = coords[0] / w
                    y1 = coords[1] / h
                    x2 = coords[2] / w
                    y2 = coords[3] / h
                    return [
                        max(0.0, min(1.0, x1)),
                        max(0.0, min(1.0, y1)),
                        max(0.0, min(1.0, x2)),
                        max(0.0, min(1.0, y2))
                    ]
            return [0.0, 0.0, 0.0, 0.0]
        except:
            return [0.0, 0.0, 0.0, 0.0]
            
    def set_generation_config(self, **kwargs):
        pass

# ==================== Evaluation Logic ====================

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'])
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en')
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'])
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    args = parser.parse_args()
    return args

def build_model(args):
    if args.model_type == "uivenus_vllm":
        model = UI_Venus_Ground_vLLM()
        model.load_model(model_name_or_path=args.model_name_or_path)
    else:
        return None
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    img_size = sample["img_size"]
    bbox_norm = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    
    if (bbox_norm[0] <= click_point[0] <= bbox_norm[2]) and (bbox_norm[1] <= click_point[1] <= bbox_norm[3]):
        return "correct"
    else:
        return "wrong"

def main(args):
    if args.task == "all":
        task_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.screenspot_test) if f.endswith(".json")]
    else:
        task_filenames = args.task.split(",")

    inst_styles = INSTRUCTION_STYLES if args.inst_style == "all" else args.inst_style.split(",")
    languages = LANGUAGES if args.language == "all" else args.language.split(",")
    gt_types = GT_TYPES if args.gt_type == "all" else args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        path = os.path.join(args.screenspot_test, task_filename + ".json")
        if not os.path.exists(path): continue
        with open(path, 'r') as f:
            task_data = json.load(f)
        
        for inst_style in inst_styles:
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        ti = copy.deepcopy(task_instance)
                        ti["task_filename"] = task_filename
                        ti["gt_type"] = gt_type
                        ti["instruction_style"] = inst_style
                        ti["language"] = lang
                        
                        if lang == "en": ti["prompt_to_evaluate"] = ti["instruction"]
                        elif lang == "cn": ti["prompt_to_evaluate"] = ti["instruction_cn"]
                        
                        if gt_type == 'positive':
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

    model = build_model(args)
    if model is None: return

    results = []
    for sample in tqdm(tasks_to_run):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        
        tmp_img = Image.open(img_path)
        img_size = tmp_img.size
        sample["img_size"] = img_size

        response = model.inference(
            instruction=sample["prompt_to_evaluate"], 
            image_path=img_path,
            img_size=img_size
        )

        pt = response["point"]
        point_px = [pt[0] * img_size[0], pt[1] * img_size[1]] if pt else None
        
        sample_result = {
            "img_path": img_path,
            "task_filename": sample["task_filename"],
            "pred": point_px,
            "raw_response": response["raw_response"],
            "bbox": sample["bbox"]
        }
        
        correctness = eval_sample_positive_gt(sample, response)
        sample_result["correctness"] = correctness
        results.append(sample_result)
        
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Finished. Saved to {args.log_path}")

if __name__ == "__main__":
    main(parse_args())