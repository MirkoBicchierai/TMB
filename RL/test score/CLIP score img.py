import numpy as np
import torch
import clip
from PIL import Image
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def load_rendered_frames(frames_directory):
    """
    Load pre-rendered frames from a directory

    Args:
        frames_directory: Path to directory containing rendered frames

    Returns:
        List of PIL Images
    """
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_directory)
                          if f.endswith(('.png', '.jpg', '.jpeg'))])

    for frame_file in frame_files:
        frame_path = os.path.join(frames_directory, frame_file)
        try:
            img = Image.open(frame_path).convert("RGB")
            frames.append(img)
        except Exception as e:
            print(f"Error loading {frame_path}: {e}")

    if not frames:
        raise ValueError(f"No valid image frames found in {frames_directory}")

    return frames


def create_motion_grid(frames, grid_size=None):
    """
    Create a grid image from multiple frames

    Args:
        frames: List of PIL Images
        grid_size: Tuple (rows, cols) for grid layout. If None, will calculate automatically.

    Returns:
        PIL Image with frames arranged in a grid
    """
    # Sample frames if there are too many
    max_frames = 500  # Maximum number of frames to include in grid
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Determine grid size
    if grid_size is None:
        grid_rows = int(np.ceil(np.sqrt(len(frames))))
        grid_cols = int(np.ceil(len(frames) / grid_rows))
    else:
        grid_rows, grid_cols = grid_size

    # Get frame dimensions
    frame_width, frame_height = frames[0].size

    # Create a blank canvas
    grid_img = Image.new('RGB', (grid_cols * frame_width, grid_rows * frame_height))

    # Paste frames into the grid
    for i, frame in enumerate(frames):
        if i >= grid_rows * grid_cols:
            break

        row = i // grid_cols
        col = i % grid_cols

        grid_img.paste(frame, (col * frame_width, row * frame_height))

    return grid_img


def get_clip_score_with_frames(frames, text):
    """
    Calculate CLIP score between a sequence of rendered frames and text

    Args:
        frames: List of PIL Images containing rendered frames
        text: Text description

    Returns:
        CLIP similarity score
    """
    # Load the pre-trained CLIP model
    model, preprocess = clip.load('ViT-B/32')

    # Create a grid from the frames
    grid_img = create_motion_grid(frames)

    # Preprocess the grid image for CLIP
    image_input = preprocess(grid_img).unsqueeze(0)
    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score, model, preprocess


def get_clip_score_for_individual_frames(frames, text, model, preprocess):
    """
    Calculate CLIP score for each individual frame

    Args:
        frames: List of PIL Images
        text: Text description
        model: Pre-loaded CLIP model
        preprocess: CLIP preprocessing function

    Returns:
        List of CLIP scores for each frame
    """
    scores = []
    text_input = clip.tokenize([text])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_input = text_input.to(device)

    for frame in frames:
        # Preprocess the frame
        image_input = preprocess(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate score
            score = torch.matmul(image_features, text_features.T).item()
            scores.append(score)

    return scores


def main():
    # Path to directory containing your rendered frames
    video_id = "3"
    frames_directory = "/home/mirko/PycharmProjects/Tesi Magistrale/ResultRL/" + video_id + "/"+video_id+"_smpl"

    text_descriptions = [
        "a person swings forward before stepping back, then puts their hand to their chest.",
        "a person waves their hands together in front of them.",
        "the person takes a few steps and starts jumping while rotating their ams.",
        "a person aggressively kneeing and kicking.",
        "a person slowly walked forward in right direction"
    ]

    frames = load_rendered_frames(frames_directory)
    print(f"Successfully loaded {len(frames)} frames")

    # Optionally save the grid visualization
    grid_img = create_motion_grid(frames)
    grid_img.save("motion_grid.png")
    print("Saved motion grid visualization to 'motion_grid.png'")

    # Calculate CLIP score for the whole sequence
    primary_text = text_descriptions[0]
    sequence_score, model, preprocess = get_clip_score_with_frames(frames, primary_text)
    print(f"\nCLIP Score for whole sequence with '{primary_text}': {sequence_score:.4f}")

    # Calculate scores for all descriptions
    print("\nScores for all descriptions:")
    for desc in text_descriptions:
        score, _, _ = get_clip_score_with_frames(frames, desc)
        print(f"CLIP Score for '{desc}': {score:.4f}")




if __name__ == '__main__':
    main()