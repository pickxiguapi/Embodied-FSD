import argparse
import base64
import math
import os
import json

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import shortuuid


# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------


def decode_image(data):
    """Convert various image representations to a RGB ``PIL.Image``."""
    # 1. bytes/bytearray
    if isinstance(data, (bytes, bytearray)):
        img_bytes = data
    # 2. base64 str
    elif isinstance(data, str):
        try:
            img_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {e}")
    # 3. numpy array
    elif isinstance(data, np.ndarray):
        if data.ndim >= 2:  # (H, W[, C]) pixel matrix
            arr = data
            if np.issubdtype(arr.dtype, np.floating):
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        elif data.ndim == 1:  # compressed byte stream
            if data.dtype == np.uint8:
                img_bytes = data.tobytes()
            else:
                img_bytes = b"".join(
                    bytes([b]) if isinstance(b, (int, np.integer)) else b for b in data
                )
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {data.shape}")
    else:
        raise ValueError(f"Unsupported image data type: {type(data)}")

    img = Image.open(BytesIO(img_bytes))
    # The model expects a 3-channel RGB image
    return img.convert("RGB")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Speed-up: disable default Torch parameter initialization
    disable_torch_init()

    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Read parquet dataset
    parquet_path = os.path.expanduser(args.parquet_file)
    df = pd.read_parquet(parquet_path)

    # Split indices into chunks
    indices = list(df.index)
    indices = get_chunk(indices, args.num_chunks, args.chunk_idx)

    # Validate mandatory columns
    if "instruction" not in df.columns:
        raise KeyError("Missing 'instruction' column in parquet file")

    image_col = None
    for cand in ["image", "images"]:
        if cand in df.columns:
            image_col = cand
            break
    if image_col is None:
        raise KeyError("Missing 'image' or 'images' column in parquet file")
    
    # Check if answer column exists
    has_answer_col = "answer" in df.columns

    # Define instruction templates based on mode
    if args.mode == "affordance_point":
        instruction_template = "You are currently a robot performing robotic manipulation tasks. Your task instruction: {instruction}. Observe the image, use 2D points and bounding box to mark the target location where the manipulated object will be moved. In your answer, use <box>[[x1, y1, x2, y2]]</box> to present the bounding box of the target region, and use <point>[[x1, y1], [x2, y2], ...]</point> to mark the points of the free space."
    elif args.mode == "visual_trace":
        instruction_template = "You are currently a robot performing robotic manipulation tasks. Your task instruction: {instruction}. Observe the image, use 2D points to mark the manipulated object-centric waypoints to guide the robot to manipulate the object.Typically, the waypoints consists of an ordered sequence of eight 2D points. The format is <point>[[x1, y1], [x2, y2], ...]</point>."
    else:
        raise ValueError(f"Unsupported mode: {args.mode}. Supported modes are 'affordance_point' and 'visual_trace'.")

    # Output file
    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:  # Only create directory if there is a directory path
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for idx in tqdm(indices):
            row = df.loc[idx]
            original_instruction = row["instruction"]
            # Apply instruction template based on mode
            instruction = instruction_template.format(instruction=original_instruction)
            img_data_raw = row[image_col]
            # If list, use the first image
            img_data = img_data_raw[0] if isinstance(img_data_raw, list) else img_data_raw

            # Build prompt
            qs = instruction
            cur_prompt = qs  # Original instruction

            if model.config.mm_use_im_start_end and model.config.mm_use_im_patch_token:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_PATCH_TOKEN * model.config.num_query_tokens
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            elif model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            # Image preprocessing
            try:
                image = decode_image(img_data)
            except Exception as e:
                print(f"[Warning] idx={idx} failed to decode image: {e}")
                continue
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

            # Generate answer
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]
            ).sum().item()
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )

            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            print("--------------------------------")
            print("question_id: ", row.get("id", idx))
            print("cur_prompt: ", cur_prompt)
            print("outputs: ", outputs)
            print("--------------------------------")
            ans_id = shortuuid.uuid()
            # Prefer 'id' column, otherwise fall back to row index
            question_id = row.get("id", idx)
            
            # Prepare output data
            output_data = {
                "question_id": question_id,
                "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {},
            }
            
            # Add ground truth answer if available
            if has_answer_col:
                output_data["ground_truth"] = row["answer"]
            
            ans_file.write(
                json.dumps(output_data, ensure_ascii=False) + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--parquet-file", type=str, default="vabench_point_dataset.parquet",
        help="Path to input parquet dataset",
    )
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--mode", type=str, default="affordance_point", 
        choices=["affordance_point", "visual_trace"],
        help="Mode for instruction template: 'affordance_point' or 'visual_trace'"
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args) 