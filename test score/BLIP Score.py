import numpy as np
import torch
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForImageTextRetrieval


def extract_frames_from_video(video_path, max_frames=100):
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


def get_blip_score_with_frames(frames, text):
    """
    Calculate BLIP score between a sequence of frames and text

    Args:
        frames: List of PIL Images containing frames
        text: Text description

    Returns:
        BLIP similarity score, model, and processor
    """

    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

    # Create a grid from the frames
    grid_img = create_motion_grid(frames)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Process the image and text
    inputs = processor(images=grid_img, text=text, return_tensors="pt").to(device)

    # Get the similarity score
    with torch.no_grad():
        outputs = model(**inputs)
        itm_score = outputs.itm_score.softmax(dim=1)[:, 1].item()  # Get the probability of image-text match

    return itm_score, model, processor

def main():
    # Path to your video file
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

    frames = extract_frames_from_video(video_path)
    print(f"Successfully extracted {len(frames)} frames from video")

    # Save the grid visualization
    grid_img = create_motion_grid(frames)
    grid_img.save("motion_grid.png")
    print("Saved motion grid visualization to 'motion_grid.png'")

    # Calculate scores for all descriptions
    print("\nScores for all descriptions:")
    for desc in text_descriptions:
        score, _, _ = get_blip_score_with_frames(frames, desc)
        print(f"BLIP Score for '{desc}': {score:.4f}")


if __name__ == '__main__':
    main()