from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import re
import ast
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

def parse_points_from_answer(answer_text: str, expected_num: int = 8) -> List[Tuple[int, int]]:
    """
    Parse the content inside <Answer></Answer> and extract points from <point> tag.
    """
    # First, extract the content inside <Answer> tag (case-insensitive)
    answer_match = re.search(r"<[Aa]nswer>(.*?)</[Aa]nswer>", answer_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        answer_content = answer_text  # If no <Answer> tag, use the original text

    # Then, extract the points from the <point> tag
    match = re.search(r"<point>\s*\[\[(.*?)\]\]\s*</point>", answer_content, re.DOTALL)
    if not match:
        raise ValueError("Could not find <point> tag or its content is empty")

    point_content = "[[" + match.group(1).strip() + "]]"

    try:
        points = ast.literal_eval(point_content)
    except Exception as e:
        raise ValueError(f"Failed to parse point coordinates: {e}\nOriginal string: {point_content}")

    assert len(points) == expected_num, f"Expected {expected_num} points, but got {len(points)}"

    return [tuple(map(int, p)) for p in points]


def convert_points_to_image_coords(
    norm_points: List[Tuple[int, int]],
    img_size: Tuple[int, int],
    norm_range: int = 1000,
) -> List[Tuple[int, int]]:
    width, height = img_size
    pad_size = max(width, height)
    scale = pad_size / norm_range

    pad_left = (pad_size - width) / 2
    pad_top = (pad_size - height) / 2

    img_points: List[Tuple[int, int]] = []
    for nx, ny in norm_points:
        x_pad = nx * scale
        y_pad = ny * scale

        x_img = x_pad - pad_left
        y_img = y_pad - pad_top

        img_points.append((int(round(x_img)), int(round(y_img))))

    return img_points


def visualize_points(
    image_path: Path,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 4,
    output_path: Path | None = None,
):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for x, y in points:
        outer_r = radius + 2
        left_up_outer = (x - outer_r, y - outer_r)
        right_down_outer = (x + outer_r, y + outer_r)
        draw.ellipse([left_up_outer, right_down_outer], fill=(255, 255, 255), outline=(255, 255, 255))

        left_up_inner = (x - radius, y - radius)
        right_down_inner = (x + radius, y + radius)
        draw.ellipse([left_up_inner, right_down_inner], fill=color, outline=color)

    if output_path is None:
        output_path = image_path.with_stem(image_path.stem + "_vis")

    img.save(output_path)
    print(f"Saved visualization to: {output_path}")

    
model_path = "/openbayes/input/input0/hf_model/FSD-v1.1-llava"
# prompt = "You are currently a robot performing robotic manipulation tasks. Your task instruction: Move the yellow block in the middle of the table. Observe the image, use 2D points to mark the manipulated object-centric waypoints to guide the robot to manipulate the object.Typically, the waypoints consists of an ordered sequence of eight 2D points. The format is <point>[[x1, y1], [x2, y2], ...]</point>."
# image_file = "./assets/image_000000.png"

prompt = "You are currently a robot performing robotic manipulation tasks. Your task instruction: Move the yellow block in the middle of the table. Observe the image, use 2D points and bounding box to mark the target location where the manipulated object will be moved. In your answer, use <box>[[x1, y1, x2, y2]]</box> to present the bounding box of the target region, and use <point>[[x1, y1], [x2, y2], ...]</point> to mark the points of the free space."
image_file = "./assets/image_000000.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024
})()

outputs = eval_model(args)
print(outputs)

# 1) 
norm_pts = parse_points_from_answer(outputs)
print(f"norm_pts: {norm_pts}")

# 2) get image size
img_file = Path(image_file)
with Image.open(img_file) as im:
    w, h = im.size

# 3) convert points to image coordinates
img_pts = convert_points_to_image_coords(norm_pts, (w, h))
print(f"img_pts: {img_pts}")

# 4) visualize points
visualize_points(img_file, img_pts) 

