"""U2Net Background Remover CLI.

A CLI based on U-2-NET default model.

Usage: python bgremover.py --model_path <model_path:u2net.pth> \
  --input_video <input-path> \
  --output_video <output-path>
"""
import os
import cv2
import torch
import numpy as np
import click
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab
from model import U2NET, U2NETP

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# Process a single frame with U^2-Net
def process_frame(frame, model, device):
    # Convert frame to PIL Image and resize
    img = Image.fromarray(frame).convert("RGB")
    img_resized = img.resize((512, 512))
    
    # Transform image to tensor
    img_tensor = torch.FloatTensor(np.array(img_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        d1, *_ = model(img_tensor)
    
    # Normalize and convert prediction to mask
    pred = normPRED(d1[:, 0, :, :]).squeeze().cpu().numpy()
    mask = (cv2.resize(pred, (frame.shape[1], frame.shape[0])) > 0.5).astype(np.uint8) * 255
    
    # Apply mask to original frame
    bg_removed = cv2.bitwise_and(frame, frame, mask=mask)
    return bg_removed

# Load and initialize model
def load_model(model_name, model_path, device):
    if model_name == "u2net":
        print("Loading U2NET (173.6 MB)...")
        model = U2NET(3, 1)
    elif model_name == "u2netp":
        print("Loading U2NETP (4.7 MB)...")
        model = U2NETP(3, 1)
    else:
        raise ValueError("Invalid model name. Choose 'u2net' or 'u2netp'.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Main function to process video
@click.command()
@click.option("--model_name", type=click.Choice(["u2net", "u2netp"]), default="u2net", help="Model name to use.")
@click.option("--model_path", type=click.Path(exists=True), required=True, help="Path to the pre-trained model file.")
@click.option("--input_video", type=click.Path(exists=True), required=True, help="Path to the input video file.")
@click.option("--output_video", type=click.Path(), default="output_video.mp4", help="Path to save the output video.")
def main(model_name, model_path, input_video, output_video):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_name, model_path, device)

    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply U2-Net to remove background
        frame_with_bg_removed = process_frame(frame, model, device)
        
        # Write the processed frame to the output video
        out.write(frame_with_bg_removed)

    # Release video objects
    cap.release()
    out.release()
    print(f"Output saved to {output_video}")

if __name__ == "__main__":
    main()
