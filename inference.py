"""
Deep Learning MCQ Solver using Qwen2-VL-7B-Instruct
Reads images containing multiple choice questions and predicts the correct option.

Usage:
    python inference.py --test_dir /path/to/test_directory

Output:
    submission.csv in the current working directory
"""

import argparse
import os
import re
import sys
import time
import pandas as pd
from PIL import Image
import torch

# ──────────────────────────────────────────────
# 1. MODEL PATH DETECTION
# ──────────────────────────────────────────────

def find_model_path():
    """
    Detect model location in order of priority:
    1. Environment variable MODEL_PATH
    2. Downloaded by setup.bash (~/models/Qwen2-VL-7B-Instruct)
    3. Kaggle model input paths
    4. HuggingFace cache (will use cached version, no download)
    """
    candidates = []

    # Env var override
    env_path = os.environ.get("MODEL_PATH", "")
    if env_path:
        candidates.append(env_path)

    # setup.bash download path
    candidates.append(os.path.expanduser("~/models/Qwen2-VL-7B-Instruct"))

    # Common Kaggle input paths
    candidates += [
        "/kaggle/input/qwen2-vl-7b-instruct",
        "/kaggle/input/qwen2-vl/transformers/qwen2-vl-7b-instruct/1",
        "/kaggle/input/qwen2vl7binstruct/transformers/default/1",
        "/kaggle/input/qwen2-vl-7b/transformers/default/1",
    ]

    for path in candidates:
        if path and os.path.isdir(path):
            # Validate it has model weights
            files = os.listdir(path)
            has_weights = any(
                f.endswith(".safetensors") or f.endswith(".bin")
                for f in files
            )
            if has_weights:
                print(f"[INFO] Model found at: {path}")
                return path

    # Last resort: HuggingFace Hub ID (uses local cache only)
    print("[INFO] Using HuggingFace Hub ID 'Qwen/Qwen2-VL-7B-Instruct' (from cache)")
    return "Qwen/Qwen2-VL-7B-Instruct"


# ──────────────────────────────────────────────
# 2. MODEL LOADING
# ──────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load model and processor.
    - bfloat16 for large GPUs (L40s 48 GB)
    - 4-bit quantization for smaller GPUs (T4 16 GB)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    if torch.cuda.is_available():
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / (1024 ** 3)
        print(f"[INFO] Total VRAM: {total_vram_gb:.1f} GB across "
              f"{torch.cuda.device_count()} GPU(s)")
    else:
        total_vram_gb = 0
        print("[WARN] No CUDA GPU detected – running on CPU (will be slow)")

    # Choose loading strategy based on available VRAM
    if total_vram_gb >= 20:
        # L40s / A100 / multi-T4: load in bfloat16 (best quality)
        print("[INFO] Loading in bfloat16 (high-VRAM mode)")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
    elif total_vram_gb >= 8:
        # Single T4 / low-VRAM: 4-bit quantization
        print("[INFO] Loading in 4-bit quantization (low-VRAM mode)")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    else:
        # CPU fallback
        print("[WARN] Loading on CPU – very slow")
        load_kwargs = {"device_map": "cpu"}

    print(f"[INFO] Loading model: {model_path} ...")
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, **load_kwargs
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print(f"[INFO] Model loaded in {time.time() - t0:.1f}s")
    return model, processor


# ──────────────────────────────────────────────
# 3. MCQ ANSWERING
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a highly knowledgeable deep learning and machine learning expert. "
    "You always reason carefully before selecting the correct answer."
)

USER_PROMPT = """Carefully examine the deep learning multiple-choice question in this image.

Follow these steps:
1. Read the question and understand exactly what is being asked.
2. Review each option (A, B, C, D) carefully.
3. Apply relevant deep learning concepts to reason through the answer.
4. Think step by step to identify the correct option.

At the very end of your response, you MUST write your final answer in EXACTLY this format (nothing else after it):
FINAL_ANSWER: [number]

Where the number must be one of:
1  →  Option A is correct
2  →  Option B is correct
3  →  Option C is correct
4  →  Option D is correct
5  →  Cannot determine (only use if truly impossible)

Example ending: FINAL_ANSWER: 2"""


def answer_mcq(model, processor, image_path: str) -> int:
    """
    Run the VLM on a single MCQ image and return an integer 1-5.
    Returns 5 (unanswered) on any error.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Cannot open image: {e}")
        return 5

    try:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=600,   # room for chain-of-thought + answer
                do_sample=False,
                repetition_penalty=1.05,
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        output_text = processor.decode(output_ids, skip_special_tokens=True).strip()

        print(f"  [DEBUG] Last 300 chars of output:\n  {output_text[-300:]}\n")
        return parse_answer(output_text)

    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        return 5


def parse_answer(text: str) -> int:
    """Extract the integer answer (1-5) from model output."""
    # Primary: look for FINAL_ANSWER: X
    match = re.search(r"FINAL_ANSWER\s*[:=]\s*([1-5])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Secondary: look for "answer is [A-D]" or "answer: [A-D]"
    letter_match = re.search(
        r"(?:answer|correct option|correct answer)\s*(?:is|:)?\s*[(\[]?\s*([A-D])\s*[)\]]?",
        text, re.IGNORECASE
    )
    if letter_match:
        return "ABCD".index(letter_match.group(1).upper()) + 1

    # Tertiary: last standalone digit 1-4 in the text
    digits = re.findall(r"\b([1-4])\b", text)
    if digits:
        return int(digits[-1])

    print("  [WARN] Could not parse answer – marking as unanswered (5)")
    return 5


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep Learning MCQ Solver")
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Absolute path to test directory (must contain test.csv and images/)",
    )
    args = parser.parse_args()

    test_dir = args.test_dir
    print(f"[INFO] Test directory: {test_dir}")

    # ── Validate directory structure ──
    test_csv_path = os.path.join(test_dir, "test.csv")
    images_dir = os.path.join(test_dir, "images")

    if not os.path.exists(test_csv_path):
        print(f"[ERROR] test.csv not found at: {test_csv_path}")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"[ERROR] images/ directory not found at: {images_dir}")
        sys.exit(1)

    test_df = pd.read_csv(test_csv_path)
    print(f"[INFO] Questions to answer: {len(test_df)}")

    # ── Load model ──
    model_path = find_model_path()
    model, processor = load_model(model_path)

    # ── Inference loop ──
    results = []
    total = len(test_df)
    start_time = time.time()

    for idx, row in test_df.iterrows():
        image_id   = row.get("image_id", row.get("image_name"))
        image_name = row["image_name"]

        # Try .png first, then no extension, then .jpg
        for ext in [".png", "", ".jpg", ".jpeg"]:
            img_path = os.path.join(images_dir, f"{image_name}{ext}")
            if os.path.exists(img_path):
                break
        else:
            img_path = None

        q_start = time.time()
        print(f"\n[{idx+1}/{total}] Image: {image_name}")

        if img_path is None:
            print(f"  [WARN] Image file not found – marking as unanswered")
            answer = 5
        else:
            answer = answer_mcq(model, processor, img_path)

        elapsed = time.time() - q_start
        total_elapsed = time.time() - start_time
        remaining = total - (idx + 1)
        eta = (total_elapsed / (idx + 1)) * remaining if idx > 0 else 0

        print(f"  → Answer: {answer}  |  Q time: {elapsed:.1f}s  |  ETA: {eta:.0f}s")

        results.append({
            "id":         image_id,
            "image_name": image_name,
            "option":     answer,
        })

    # ── Save submission ──
    # IMPORTANT: saved in current working directory, NOT test_dir
    submission_df = pd.DataFrame(results)
    output_path = "submission.csv"
    submission_df.to_csv(output_path, index=False)

    total_time = time.time() - start_time
    print(f"\n[INFO] ✓ Submission saved → {os.path.abspath(output_path)}")
    print(f"[INFO] Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n{submission_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
