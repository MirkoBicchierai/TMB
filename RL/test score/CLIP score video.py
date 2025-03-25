import numpy as np
import torch
import clip
from PIL import Image
import cv2


def extract_frames_from_video(video_path, max_frames=36):
    """
    Extract frames from a video file

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract

    Returns:
        List of PIL Images
    """
    frames = []

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")

    # Calculate frame indices to extract
    if total_frames <= max_frames:
        # Extract all frames if there are fewer than max_frames
        frame_indices = list(range(total_frames))
    else:
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for frame_idx in frame_indices:
        # Set video position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        success, frame = video.read()
        if not success:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb)
        frames.append(pil_frame)

    # Release the video capture object
    video.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")

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
    Calculate CLIP score between a sequence of frames and text

    Args:
        frames: List of PIL Images containing frames
        text: Text description

    Returns:
        CLIP similarity score, model, and preprocess function
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
    # Path to your video file
    video_id = "3"
    type = True

    if type:
        type_str = "joints"
    else:
        type_str = "smpl"

    video_path = "/home/mirko/PycharmProjects/Tesi Magistrale/ResultRL/" + video_id + "/" + video_id + "_"+type_str+".mp4"

    text_descriptions = [
        "a person swings forward before stepping back, then puts their hand to their chest.",
        "a person waves their hands together in front of them.",
        "the person takes a few steps and starts jumping while rotating their ams.",
        "a person aggressively kneeing and kicking.",
        "a person slowly walked forward in right direction"
    ]

    frames = extract_frames_from_video(video_path)
    print(f"Successfully extracted {len(frames)} frames from video")

    # Save the grid visualization
    grid_img = create_motion_grid(frames)
    grid_img.save("motion_grid.png")
    print("Saved motion grid visualization to 'motion_grid.png'")

    # Calculate scores for all descriptions
    print("\nScores for all descriptions:")
    for desc in text_descriptions:
        score, _, _ = get_clip_score_with_frames(frames, desc)
        print(f"CLIP Score for '{desc}': {score:.4f}")


if __name__ == '__main__':
    main()