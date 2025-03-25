import cv2
import torch
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel

# Load model and processor
model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def process_video(video_path, num_frames=8):
    """Extract and select frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Select evenly spaced frames
    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        selected_frames = [frames[i] for i in indices]
    else:
        # Handle short videos by padding with last frame
        selected_frames = frames + [frames[-1]] * (num_frames - len(frames))

    return selected_frames


def calculate_similarity(video_path, text_query):
    """Calculate similarity score between video and text"""
    # Process video
    video_frames = process_video(video_path)

    # Prepare inputs
    inputs = processor(
        text=[text_query],
        videos=[video_frames],  # Wrap in list for single video
        return_tensors="pt",
        padding=True
    ).to(device)

    # Calculate similarity
    with torch.no_grad():
        outputs = model(**inputs)

    # Return similarity score
    return outputs.logits_per_video.item()


# Example usage
video_path = "/home/mirko/PycharmProjects/Tesi Magistrale/ResultRL/1/1_smpl.mp4"
text_query = "a person walks up stairs."
similarity_score = calculate_similarity(video_path, text_query)

print(f"Similarity score between video and text: {similarity_score:.4f}")