import os
import cv2
import numpy as np
import argparse
import torch
from torchvision.transforms import Compose
import matplotlib
from SEEDepth_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F

def get_transform(width, height, resize_target=False):
    transform = Compose([
        Resize(
            width=width,
            height=height,
            resize_target=resize_target,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return transform

def process_images_with_model(input_folder, output_folder, model_path, model_version="v2", model_size="large"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    output_folder_raw = os.path.join(output_folder, "npy")
    output_folder_normalized = os.path.join(output_folder, "normalized_images")
    if not os.path.exists(output_folder_raw):
        os.makedirs(output_folder_raw)
        print(f"Created output folder: {output_folder_raw}")
        
    if not os.path.exists(output_folder_normalized):
        os.makedirs(output_folder_normalized)
        print(f"Created output folder: {output_folder_normalized}")

    checkpoint_name = f"SEEDepth_{model_version}_{model_size}.pth"
    restore_ckpt = os.path.join(model_path, checkpoint_name)
    if model_version == "v2":
        if model_size == "large":
            from SEEDepth_v2.SEEDepth_v2_large import SEEDepth_v2_large as SEEDepth
        elif model_size == "base": 
            from SEEDepth_v2.SEEDepth_v2_base import SEEDepth_v2_base as SEEDepth
        else:
            raise ValueError("Invalid model size. Choose 'large' or 'base'.")
    elif model_version == "v1":
        if model_size == "large":
            from SEEDepth_v1.SEEDepth_v1_large import SEEDepth_v1_large as SEEDepth
        elif model_size == "base":
            from SEEDepth_v1.SEEDepth_v1_base import SEEDepth_v1_base as SEEDepth
        else:
            raise ValueError("Invalid model size. Choose 'large' or 'base'.")
    else:
        raise ValueError("Invalid model version. Choose 'v1' or 'v2'.")
    model = SEEDepth().to(device)
    checkpoint = torch.load(restore_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()

    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    if not png_files:
        print(f"No images found in: {input_folder}")
        return

    print(f"Found {len(png_files)} images in {input_folder}. Starting processing...")

    cmap = matplotlib.colormaps.get_cmap('magma')
    for filename in png_files:
        input_image_path = os.path.join(input_folder, filename)
        output_raw_image_path = os.path.join(output_folder_raw, f"disparity_{filename.replace('.png', '.npy')}")
        output_normalized_image_path = os.path.join(output_folder_normalized, f"disparity_{filename}")

        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Warning: Could not load image: {input_image_path}. Skipping.")
            continue

        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = processed_image.shape[:2]
        transform = get_transform(w, h, resize_target=False)
        processed_image = transform({'image': processed_image})['image']
        processed_image = torch.from_numpy(processed_image).unsqueeze(0).to(device)

        with torch.no_grad():
            disparity = model(processed_image)

        disparity = F.interpolate(disparity, (h, w), mode='bilinear', align_corners=False)[0, 0]
        normalize_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255.0


        disparity = disparity.cpu().numpy()

        normalize_disparity = normalize_disparity.cpu().numpy().astype(np.uint8)
        normalize_disparity = (cmap(normalize_disparity)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        np.save(output_raw_image_path, disparity)
        cv2.imwrite(output_normalized_image_path, normalize_disparity)
        print(f"Processed '{filename}' and saved to '{output_raw_image_path}'")
        print(f"Processed '{filename}' and saved to '{output_normalized_image_path}'")


    print("Image processing complete!")


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('--input_directory', type=str, default='./input_images',
                        help='Path to the folder containing input JPG images')
    parser.add_argument('--output_directory', type=str, default='./output_preds',
                        help='Path to the folder where prediction of disparity will be saved')
    parser.add_argument('--checkpoints_directory', type=str, default='/home/sanggyun/datasets/SEEDepth_checkpoints',
                        help='Path to the directory where model checkpoints are stored')
    parser.add_argument('--model_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='Version of the model to use (default: v2, choices: v1, v2)')
    parser.add_argument('--model_size', type=str, default='large', choices=['base', 'large'],
                        help='Size of the model to use (default: large, choices: base, large)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the processing function with the parsed arguments
    process_images_with_model(
        input_folder=args.input_directory,
        output_folder=args.output_directory,
        model_path=args.checkpoints_directory,
        model_version=args.model_version,
        model_size=args.model_size
    )