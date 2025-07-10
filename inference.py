import os
import cv2
import numpy as np
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

# --- How to use the function ---
if __name__ == "__main__":
    # Define your input and output folders
    input_directory = "./input_images"
    output_directory = "./output_preds"
    checkpoints_directory = "/home/sanggyun/datasets/SEEDepth_checkpoints"
    model_version = "v2" # or "v1"
    model_size = "large" # or "base"

    process_images_with_model(input_directory, output_directory, checkpoints_directory,model_version, model_size)