import base64
import json
import os
import re
from copy import deepcopy
from typing import Dict, List

import cv2
import jsonlines
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.interpolate import interp1d
from tqdm import tqdm

prompt_template = """You are an expert evaluator in robotic manipulation and visual reasoning. Your job is to assess the quality of predicted trajectories based on task instructions and visual inputs.

You are given:
- A task instruction describing an object manipulation task.
- An image showing a predicted trajectory.

**Note:**
- In the image, the red circle indicates the start point, and the blue diamond indicates the end point.
- The trajectory represents the predicted movement path of the manipulated object, not the robot or end-effector.
- You should **evaluate the predicted trajectory as a proposed motion for the object that is supposed to be moved**, based on the task instruction — **not based on the static positions of objects in the image**. The objects have not actually moved.

**Evaluation Criteria (listed in order of importance):**

1. **Task Alignment and Success (most important)**  
   - Does the trajectory clearly and accurately fulfill the task instruction?  
   - **The trajectory must start at the correct location and end at a target position that aligns with the task goal.**  
   - Large deviations in the starting or ending point (e.g., wrong object, wrong destination, or stopping short of the goal) should result in a low score, even if the rest of the trajectory is smooth.  
   - If the task is not accomplished (due to incorrect goal interpretation or spatial execution), the score should be low regardless of other qualities.

2. **Feasibility**  
   - Is the movement physically plausible, smooth, and continuous?  
   - Are there any unrealistic discontinuities, sharp turns, or impossible transitions?  
   - Even if the movement is feasible, it should not receive a high score if the task is not completed.

3. **Obstacle Avoidance / Safety**  
   - Does the trajectory reasonably avoid collisions with surrounding objects?  
   - Minor risks may be tolerated if the task is completed successfully, but major or clear collisions should reduce the score.

**Scoring Guideline:**
- If the task is **not accomplished**, or if the start or end point is significantly incorrect, the score should typically be **4 or below**.
- If the task is completed but the trajectory has issues (e.g., roughness, minor risk of collision), a score in the **6–8** range is appropriate.
- A **score of 9–10** should be given only when the trajectory clearly completes the task, with good start/end accuracy, smooth motion, and reasonable safety.

Based on these criteria, provide a single overall score from 1 (very poor) to 10 (excellent), reflecting how well the task is accomplished.

Respond strictly in the following format:
Score: <1-10>  
Explanation: <brief justification>

The task instruction is:  
{task_instruction}

Please give your response.
"""

API_KEY = "Your API Key"

class OpenAIClient:
    def __init__(
        self, api_key: str = "sk-", base_url: str = ""
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content


def interpolate_trajectory(traj, new_length):
    traj = np.array(traj)
    x = np.linspace(0, 1, len(traj))
    f = interp1d(x, traj, axis=0, kind="linear")
    x_new = np.linspace(0, 1, new_length)
    interpolated_traj = f(x_new).tolist()
    return interpolated_traj


def visualize_trajectories(
    image,
    pred_traj,
    ans_traj,
    instruction,
    filename="trajectory_visualization.png",
    show_ground_truth=False,
):
    image = image.copy()
    height, width, _ = image.shape

    for i in range(1, len(pred_traj)):
        ratio = i / (len(pred_traj) - 1)
        r = int(139 * (1 - ratio))
        b = int(139 * ratio)
        color = (b, 0, r)
        x1, y1 = (
            int(pred_traj[i - 1][0] / 1000 * width),
            int(pred_traj[i - 1][1] / 1000 * height),
        )
        x2, y2 = (
            int(pred_traj[i][0] / 1000 * width),
            int(pred_traj[i][1] / 1000 * height),
        )
        cv2.line(image, (x1, y1), (x2, y2), color, 3)

    x0, y0 = int(pred_traj[0][0] / 1000 * width), int(pred_traj[0][1] / 1000 * height)
    cv2.circle(image, (x0, y0), 9, (255, 255, 255), 2)
    cv2.circle(image, (x0, y0), 6, (0, 0, 139), -1)

    x_end, y_end = (
        int(pred_traj[-1][0] / 1000 * width),
        int(pred_traj[-1][1] / 1000 * height),
    )
    diamond_size = 10
    diamond_pts = np.array(
        [
            [x_end, y_end - diamond_size],
            [x_end + diamond_size, y_end],
            [x_end, y_end + diamond_size],
            [x_end - diamond_size, y_end],
        ],
        np.int32,
    )
    cv2.polylines(image, [diamond_pts], True, (255, 255, 255), 5)
    cv2.fillPoly(image, [diamond_pts], (139, 0, 0))

    if show_ground_truth:
        for i in range(1, len(ans_traj)):
            ratio = i / (len(ans_traj) - 1)
            r = int(139 * (1 - ratio))
            b = int(139 * ratio)
            color = (b, 0, r)
            x1, y1 = (
                int(ans_traj[i - 1][0] / 1000 * width),
                int(ans_traj[i - 1][1] / 1000 * height),
            )
            x2, y2 = (
                int(ans_traj[i][0] / 1000 * width),
                int(ans_traj[i][1] / 1000 * height),
            )
            cv2.line(image, (x1, y1), (x2, y2), color, 3)

        x0, y0 = int(ans_traj[0][0] / 1000 * width), int(ans_traj[0][1] / 1000 * height)
        cv2.circle(image, (x0, y0), 9, (255, 255, 255), 3)
        cv2.circle(image, (x0, y0), 6, (0, 0, 139), -1)

        x_end, y_end = (
            int(ans_traj[-1][0] / 1000 * width),
            int(ans_traj[-1][1] / 1000 * height),
        )
        diamond_size = 10
        diamond_pts = np.array(
            [
                [x_end, y_end - diamond_size],
                [x_end + diamond_size, y_end],
                [x_end, y_end + diamond_size],
                [x_end - diamond_size, y_end],
            ],
            np.int32,
        )
        cv2.polylines(
            image, [diamond_pts], True, (255, 255, 255), 3
        )
        cv2.fillPoly(image, [diamond_pts], (139, 0, 0))

    image_vis = deepcopy(image)
    text = instruction
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30

    rectangle_bgr = (200, 200, 200)
    padding = 5
    cv2.rectangle(
        image_vis,
        (text_x - padding, text_y + padding),
        (text_x + text_size[0] + padding, text_y - text_size[1] - padding),
        rectangle_bgr,
        -1,
    )

    cv2.putText(
        image_vis, instruction, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
    )

    cv2.imwrite(filename, image_vis)

    return image


results_file = (
    "gpt_score_example.jsonl"
)
parquet_file_path = "test.parquet"
visualize = True  # True to enable visualization, False to disable
show_ground_truth = False  # True to show ground truth trajectory, False to not show

df = pd.read_parquet(parquet_file_path)

with jsonlines.open(results_file, "r") as f:
    results = list(f)

outputs = []
for record in tqdm(results):
    task_instruction = record["doc"]["problem"].split(":")[1].split(".")[0].strip()
    doc_id = record["doc"]["id"]
    row = df[df["id"] == doc_id]
    image_bytes = row["images"].iloc[0][0]
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    height, width, _ = image.shape
    filtered_resps = str(record["filtered_resps"][0])
    answer_match = re.search(r"<Answer>(.*?)</Answer>", filtered_resps, re.DOTALL)
    answer_content = answer_match.group(1)
    point_match = re.search(r"<point>(.*?)</point>", answer_content)
    pred_traj_str = point_match.group(1)
    pred_traj = json.loads(pred_traj_str)

    answer = record["doc"]["answer"]
    answer = answer.replace("<type>fsd_visual_trace</type>", "")
    json_answer = json.loads(answer)
    ans_traj = json_answer["trajectory"]

    # Pad to square
    max_side = max(height, width)
    pad_top = (max_side - height) // 2
    pad_bottom = max_side - height - pad_top
    pad_left = (max_side - width) // 2
    pad_right = max_side - width - pad_left

    # Scale and normalize trajectory points
    ans_traj_scaled = []
    for point in ans_traj:
        x, y = point
        x = (x / width) * (1000 - 1)
        y = (y / height) * (1000 - 1)
        ans_traj_scaled.append([x, y])
    ans_traj = ans_traj_scaled

    # Interpolate to the longer trajectory length
    new_length = max(len(pred_traj), len(ans_traj))
    pred_traj_interp = interpolate_trajectory(pred_traj, new_length)
    ans_traj_interp = interpolate_trajectory(ans_traj, new_length)

    # Visualize and save the trajectory
    output_dir = "visualized_trajectories_fsd"  # Specify the save folder
    os.makedirs(output_dir, exist_ok=True)  # Ensure the folder exists
    output_path = os.path.join(output_dir, f"{doc_id}.png")  # Use doc_id as the file name
    img_input = visualize_trajectories(
        image,
        pred_traj_interp,
        ans_traj_interp,
        task_instruction,
        output_path,
        show_ground_truth,
    )
    _, buffer = cv2.imencode(".jpg", img_input)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    client = OpenAIClient(
        api_key=API_KEY
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template.format(task_instruction=task_instruction),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]

    response = client.chat_completion(
        model="gpt-4.1", messages=messages, temperature=0.0
    )
    print(response)
    outputs.append(
        {"id": doc_id, "task_instruction": task_instruction, "response": response}
    )

with open("vabench_visual_trace_gpt_eval.jsonl", "w") as f:
    for output in outputs:
        f.write(json.dumps(output) + "\n")


with jsonlines.open(
    "vabench_visual_trace_gpt_eval.jsonl", "r"
) as f:
    data = list(f)

avg_scores = []
for item in data:
    response = item["response"]
    score = re.search(r"Score: (\d+)", response)
    if score:
        score = int(score.group(1))
        avg_scores.append(score)
    else:
        print(response)

print(sum(avg_scores) / len(avg_scores))
