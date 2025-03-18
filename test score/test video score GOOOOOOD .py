import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import clip

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def extract_video_frames(video_path, num_frames=120):
    """
    Extract evenly spaced frames from a video.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = [Image.fromarray(vr[i].asnumpy()) for i in frame_indices]  # Convert to PIL images
    return frames

def compute_similarity(video_path, text_query):
    """
    Compute the similarity between a text query and a video.
    """
    # Extract and preprocess video frames
    frames = extract_video_frames(video_path)
    frame_tensors = [preprocess(frame).unsqueeze(0).to(device) for frame in frames]
    frame_tensors = torch.cat(frame_tensors)  # Shape: (num_frames, 3, 224, 224)

    # Encode video frames with CLIP
    with torch.no_grad():
        video_features = model.encode_image(frame_tensors)  # Shape: (num_frames, 512)
        video_features = video_features.mean(dim=0, keepdim=True)  # Average frame embeddings

    # Encode text with CLIP
    text_inputs = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)  # Shape: (1, 512)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(video_features, text_features)
    return similarity.item()

# Example usage
video_id = "3"
type = False

if type:
    type_str = "joints"
else:
    type_str = "smpl"

video_path = "/home/mirko/PycharmProjects/Tesi Magistrale/ResultRL/" + video_id + "/" + video_id + "_" + type_str + ".mp4"

text_descriptions = [
    "a person swings forward before stepping back, then puts their hand to their chest.",
    "a person waves their hands together in front of them.",
    "the person takes a few steps and starts jumping while rotating their ams.",
    "a person aggressively kneeing and kicking.",
    "a person slowly walked forward in right direction"
]

for text_description in text_descriptions:
    score = compute_similarity(video_path, text_description)
    print(f"Similarity score: {text_description}, {score:.4f}")
