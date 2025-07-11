# SEEDepth

This work presents Semantic-Enhanced and Efficient Distillation for Depth Estimation (SEEDepth), a two-stage framework that effectively fuses information of depth and semantic foundation models to enhance Monocular Depth Estimation (MDE).

## Usage

### Installation (Conda environment)

- Python `3.11.7`

```bash
git clone https://github.com/wodon326/SEEDepth.git
cd SEEDepth
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Models

We provide two models of varying scales for relative depth estimation based on [DepthAnything-V1](https://github.com/LiheYoung/Depth-Anything) and [DepthAnything-V2](https://github.com/DepthAnything/Depth-Anything-V2):

| Model | # Params |
| :---: | :---: |
| DepthAnything-V1-Base | 98M |
| [DepthAnything-V1-Base + SEEDepth](https://drive.google.com/uc?export=download&id=1uQFZF0n_93Kem27dOOngtWhBVaB9xTIy) | 105M |
| DepthAnything-V1-Large | 335M |
| [DepthAnything-V1-Large + SEEDepth](https://drive.google.com/uc?export=download&id=1aYSt1aqgXC54LqmcZPdN_fS2A7EY-NyK) | 349M |
| DepthAnything-V2-Base | 98M |
| [DepthAnything-V2-Base + SEEDepth](https://drive.google.com/uc?export=download&id=1x0VXkMDeeZEmD71_5qO3d7VwOagaVUCs) | 105M |
| DepthAnything-V2-Large | 335M |
| [DepthAnything-V2-Large + SEEDepth](https://drive.google.com/uc?export=download&id=1oAyGVpQAPUd4YUCyPBfJ3mtiwnmxDuWi) | 349M |

To run the model, you need download each model above.

### Run

```python
python infer.py \
    --input_root_path ~/data/AsymKD/scannet_val_sampled_800_1 \
    --input_filename_path evaluation/data_split/scannet/scannet_val_sampled_list_800_1.txt \
    --outdir evaluation/output/scannet/briges_depth_anythingv2 \
    --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
    --encoder vitl \
    --infer_width 686 \
    --infer_height 518
```

Arguments:

- `--model`
- `--img_path`
