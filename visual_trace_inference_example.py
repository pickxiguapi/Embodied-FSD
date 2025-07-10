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
    
    answer_match = re.search(r"<[Aa]nswer>(.*?)</[Aa]nswer>", answer_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        answer_content = answer_text

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


def visualize_visual_trace(
    image_path: Path,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 6,
    line_width: int = 3,
    output_path: Path | None = None,
):
    """Draw a smooth trajectory on the image based on given 2D pixel coordinates.

    The function applies Catmull-Rom spline interpolation over the provided
    waypoints, draws the resulting curve, marks the first point with a filled
    square, and the last point with a filled diamond.
    """

    import math

    if len(points) < 2:
        raise ValueError("points list must contain at least two coordinates")

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    
    def catmull_rom(p0, p1, p2, p3, n_samples: int = 20):
        curve = []
        for i in range(n_samples):
            t = i / (n_samples - 1)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            curve.append((x, y))
        return curve

    smooth_pts = []
    n = len(points)
    for i in range(n - 1):
        p0 = points[i - 1] if i - 1 >= 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]
        segment = catmull_rom(p0, p1, p2, p3, n_samples=20)
        
        smooth_pts.extend(segment)

    
    draw.line(smooth_pts, fill=color, width=line_width, joint="curve")

    
    sx, sy = points[0]
    draw.rectangle(
        [(sx - radius, sy - radius), (sx + radius, sy + radius)],
        fill=color,
        outline=(255, 255, 255),
    )

    
    ex, ey = points[-1]
    diamond = [
        (ex, ey - radius),
        (ex + radius, ey),
        (ex, ey + radius),
        (ex - radius, ey),
    ]
    draw.polygon(diamond, fill=color, outline=(255, 255, 255))

    if output_path is None:
        output_path = image_path.with_stem(image_path.stem + "_vis")

    img.save(output_path)
    print(f"Saved visualization to: {output_path}")

    
model_path = "/openbayes/input/input0/hf_model/FSD-v1.1-llava"
prompt = "You are currently a robot performing robotic manipulation tasks. Your task instruction: Put carrot on plate. Observe the image, use 2D points to mark the manipulated object-centric waypoints to guide the robot to manipulate the object.Typically, the waypoints consists of an ordered sequence of eight 2D points. The format is <point>[[x1, y1], [x2, y2], ...]</point>."
image_file = "./assets/image_000008.png"

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

 
norm_pts = parse_points_from_answer(outputs)
print(f"norm_pts: {norm_pts}")

 
img_file = Path(image_file)
with Image.open(img_file) as im:
    w, h = im.size

 
img_pts = convert_points_to_image_coords(norm_pts, (w, h))
print(f"img_pts: {img_pts}")

 
visualize_visual_trace(img_file, img_pts) 

