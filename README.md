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

To run the model, you need to download each model above.

### Run

```bash
python inference.py \
    --input_dir <input-dir> \
    --output_dir <output-dir> \
    --checkpoints_dir <checkpoints-dir> \
    --model_version <v1 | v2> \
    --model_size <base | large>
```

Arguments:

- `--input_dir`: Path to the folder containing input images (default: `./input_images`)
- `--output_dir`: Path to the folder where prediction of disparity will be saved (default: `./output_preds`)
- `--checktpoints_dir`: Path to the directory where model checkpoints are stored (default: `./checkpoints`)
- `--model_version`: Version of the model to use (default: `v2`)
- `--model_size`: Size of the model to use (default: `large`)

### Acknowledgements

Thanks to these amazing open source projects:
- [DepthAnything-V1](https://github.com/LiheYoung/Depth-Anything)
- [DepthAnything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [SegmentAnything](https://github.com/facebookresearch/segment-anything)
- [Marigold](https://github.com/prs-eth/Marigold)
- Sample images in `input_images` are sampled from [DIODE](https://diode-dataset.org/) dataset.
