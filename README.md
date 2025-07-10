# From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation (Embodied-FSD)

<div align="center">

**From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation**

[[🌐 Website](https://embodied-fsd.github.io)] [[📄 Paper](https://arxiv.org/pdf/2505.08548)] [[🤗 Models](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)] [[🎯 Datasets](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)] [[💬 Demo](#demo)]

</div>

---

## 📖 Introduction 

We present **FSD (From Seeing to Doing)** with:

***[Embodied-FSD Model](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)***: We develop FSD, a novel vision-language model that generates intermediate representations through spatial relationship reasoning, providing fine-grained guidance for robotic manipulation. It integrates Spatial Relationship-Focused Chain-of-Thought (Sr-CoT) reasoning while maintaining powerful general capabilities.

***[VABench](https://huggingface.co/collections/IffYuan/fsd-683fa0d552e70f302fd04b34)***: We propose VABench, a more challenging benchmark for evaluating visual aids generation capabilities in robotic manipulation scenarios.

<img width="800" alt="image" src="assets/framework.jpg">

Figure 1: Overview of FSD

<img width="800" alt="image" src="assets/srcot.jpg">

Figure 2: Spatial relationship-focused reasoning process (SrCoT).

## 📰 News

- **[2025-07]** 🔬 We have updated the detailed training, inference, and evaluation code and readme. VABench evaluation benchmark is officially released!
- **[2025-05]** 📝 Code repository is now public - welcome to try FSD for robotic manipulation!

## ⚙️ Setup (Same as LLaVA)

1. Clone this repository and navigate to Embodied-FSD folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# we recommend transformers==4.31.0
pip install transformers==4.31.0
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

---

## 🚀 Inference 

### Affordance Point Example

Task instruction: Move the yellow block in the middle of the table.

**Before prediction (original image):**

<img src="assets/image_000000.png" width="400" alt="Original input image">

**Run the example code:**
```bash
cd Embodied-FSD/
python affordance_point_inference_example.py
```

**After prediction (visualization result):**

<img src="assets/image_000000_vis.png" width="400" alt="Visualization result with predicted points">

### Visual Trace Example

Task instruction: put carrot on plate.

**Before prediction (original image):**

<img src="assets/image_000008.png" width="400" alt="Original input image">

**Run the example code:**
```bash
cd Embodied-FSD/
python visual_trace_inference_example.py
```

**After prediction (visualization result):**

<img src="assets/image_000008_vis.png" width="400" alt="Visualization result with predicted points">

---

## 🎯 Training

We mainly use the LLaVA and ASMv2 codebases to develop FSD. We appreciate these excellent works. The training process of FSD is divided into two stages: the first stage focuses on embodied reasoning and general spatial reasoning, while the second stage focuses on visual aids generation.

### Data Preparation

Please download the required datasets and organize them in `./data` from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), [train2014](http://images.cocodataset.org/zips/train2014.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- CLEVR_v1.0: [images](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip)
- Visual7W: [images](http://vision.stanford.edu/yukezhu/visual7w_images.zip)
- Flickr30K: [images](https://hockenmaier.cs.illinois.edu/DenotationGraph/)
- SA-1B: [images](https://ai.meta.com/datasets/segment-anything/) (Only sa_000000-sa_000003)
- st_vqa(cauldron,llava_format), raven(cauldron), vsr(cauldron,llava_format), CLEVR-Math(MathV360K), Super-CLEVR(MathV360K): [FSD-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/tree/main) (derived from LLaVA-OneVision-Data)
- kitti, 2d3ds: [FSD-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/tree/main) (derived from SpatialQA)  
- object_ref, region_ref: [FSD-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/tree/main) (derived from RoboPoint)  
- bridge_data_v2: [images](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/) (derived from BridgeDataV2)
- droid: [FSD-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/tree/main) (derived from DROID)  
- rtx: [FSD-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/tree/main) (derived from Open-Embodidedment-X)  

After downloading all datasets, organize the data as follows in `./data`:

```
├── coco
│   ├── train2014
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
│   ├── VG_100K
│   └── VG_100K_2
├── CLEVR_v1.0
│   └── images
├── Visual7W
│   └── images
├── flickr30k
│   └── images
├── sam
│   ├── sa_000000
│   ├── sa_000001
│   ├── sa_000002
│   └── sa_000003
├── st_vqa(cauldron,llava_format)
├── raven(cauldron)
├── vsr(cauldron,llava_format)
├── CLEVR-Math(MathV360K)
├── Super-CLEVR(MathV360K)
├── SAT_images
├── kitti
├── 2d3ds
├── bridge_data_v2
│   ├── bridge_data_v1
│   ├── bridge_data_v2
│   ├── flap
│   ├── rss
│   └── icra 
├── droid  
│   ├── ILIAD+j807b3f8+2023-05-11-17h-34m-39s
│   └── ...
├── rtx 
│   ├── fractal20220817_data
│   ├── ucsd_kitchen_dataset_converted_externally_to_rlds
│   ├── jaco_play 
│   └── ucsd_pick_and_place_dataset_converted_externally_to_rlds         
├── object_ref
├── region_ref
```

### Stage1: General Embodied/Spatial Reasoning

In this stage, we train the model to enhance spatial reasoning ability. We finetune the FSD model based on the [ASMv2](https://github.com/OpenGVLab/all-seeing).

The JSON data used in Stage 1: [Dataset Link](https://huggingface.co/datasets/IffYuan/FSD-Dataset/blob/main/data/FSD-Stage1-Dataset.json)

```shell
# Stage 1: spatial reasoning
bash scripts_fsd/stage1-fsd.sh
```

### Stage2: Robotics-Focused Fine-tuning

In this stage, we enhance the model with robotics manipulation data and advanced visual aids generation.

The JSON data used in Stage 2: [Dataset Link](https://huggingface.co/datasets/IffYuan/FSD-Dataset/blob/main/data/FSD-Stage2-Dataset.json)

```shell
# Stage 2: visual aids generation
bash scripts_fsd/stage2-fsd.sh
```

## 📊 Weak-to-Strong Dataset

- [Level-1-2-3-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/blob/main/data/Level-1-2-3-Dataset.json): FSD spatial reasoning dataset
- [Level-4-5-Dataset](https://huggingface.co/datasets/IffYuan/FSD-Dataset/blob/main/data/Level-4-5-Dataset.json): FSD visual aids generation dataset

As in ASMv2, in our dataset, we use <ref></ref> to annotate target objects and <pred></pred> to annotate spatial relations. Each bounding box is normalized to integer values within the range [0, 1000). **Note: When training and outputting coordinates, we first pad the image into a square and then output the normalized coordinates on the square image. Special attention should be paid to this conversion process.**

## 📝 Evaluation

We used the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework to complete the evaluation of all benchmarks, and we are grateful for their outstanding work!

### VABench Evaluation

`vabench_point_dataset.parquet` and `vabench_visual_trace_dataset.parquet` are used for VABench-Point and VABench-Visual Trace, respectively. In the parquet files, the `instruction` column contains the task instructions, the `images` column contains the images, and the `answer` column contains the answers. When evaluating VABench-Point, we calculate the proportion of predicted points that fall within the answer bounding boxes as the accuracy. For VABench-Visual Trace, we compute the MAE and RMSE between the predicted trajectories and the ground-truth trajectories. Note that, to ensure fair comparison across images of different sizes, both the predicted results and the ground-truth results are converted to the 0-1000 normalized coordinate system of the padded images (since FSD predictions are already in this format, no conversion is needed for them).

*The code for evaluation using GPT is to be updated.*

### SIMPLERENV Evaluation

Clean code ing...Coming soon...

---

## 🙏 Acknowledgments

We sincerely thank the following outstanding open-source projects and research works, which have provided an important foundation and support for the development of FSD: 

- [LLaVA](https://github.com/haotian-liu/LLaVA)  
- [ASMv2](https://github.com/OpenGVLab/all-seeing)  
- [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)   
- [SpatialQA](https://github.com/Junjie-Ye/SpatialQA)  
- [RoboPoint](https://github.com/huijieZH/RoboPoint)  
- [BridgeDataV2](https://rail.eecs.berkeley.edu/datasets/bridge_release/)  
- [DROID](https://droid-dataset.github.io/)  
- [Open-X-Embodiment](https://robotics-transformer-x.github.io/)  

---

## 📜 License

This project is licensed under the Apache 2.0 License. For details, please see the [LICENSE](LICENSE) file.

---

## 📚 Citation

If you use FSD in your research, please cite our paper:
```
@misc{yuan2025seeingdoingbridgingreasoning,
      title={From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation}, 
      author={Yifu Yuan and Haiqin Cui and Yibin Chen and Zibin Dong and Fei Ni and Longxin Kou and Jinyi Liu and Pengyi Li and Yan Zheng and Jianye Hao},
      year={2025},
      eprint={2505.08548},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.08548}, 
}
```