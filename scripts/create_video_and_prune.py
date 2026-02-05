#!/usr/bin/env python
"""
Script to create a video from images in a dataset folder with frame numbers displayed,
and then create a pruned dataset based on specified frame ranges.

Usage:
    # Step 1: Create video to review frames
    python create_video_and_prune.py --folder train_3 --mode video
    
    # Step 2: After reviewing video, create pruned dataset
    python create_video_and_prune.py --folder train_3 --mode prune --keep-ranges "0-100,150-300,400-500"
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
import re

def parse_frame_number(filename):
    """Extract frame number from filename like '000000-0.50.jpg'"""
    # Frame number is the first 6 digits
    match = re.match(r'(\d{6})', filename)
    if match:
        return int(match.group(1))
    return None

def parse_angle(filename):
    """Extract angle from filename like '000000-0.50.jpg'"""
    # Angle is after the frame number
    match = re.match(r'\d{6}([-]?\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def create_video_with_frame_numbers(folder_name, output_video=None, fps=10):
    """
    Create a video from images in the specified folder with frame numbers displayed.
    
    Args:
        folder_name: Name of the folder containing images (e.g., 'train_3')
        output_video: Path to output video file (default: data/{folder_name}_review.mp4)
        fps: Frames per second for the output video
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_path, "..", "data", folder_name)
    
    if not os.path.exists(data_folder):
        print(f"Error: Folder '{data_folder}' does not exist.")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
    
    if len(image_files) == 0:
        print(f"Error: No images found in '{data_folder}'")
        return
    
    # Parse and sort by frame number
    image_data = []
    for img_file in image_files:
        frame_num = parse_frame_number(img_file)
        angle = parse_angle(img_file)
        if frame_num is not None:
            image_data.append({
                'filename': img_file,
                'frame_num': frame_num,
                'angle': angle
            })
    
    image_data = sorted(image_data, key=lambda x: x['frame_num'])
    
    if len(image_data) == 0:
        print(f"Error: No valid images found in '{data_folder}'")
        return
    
    print(f"Found {len(image_data)} images")
    print(f"Frame range: {image_data[0]['frame_num']} to {image_data[-1]['frame_num']}")
    
    # Set output video path
    if output_video is None:
        output_video = os.path.join(script_path, "..", "data", f"{folder_name}_review.mp4")
    
    # Read first image to get dimensions
    first_img_path = os.path.join(data_folder, image_data[0]['filename'])
    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Creating video at {fps} FPS...")
    print(f"Output: {output_video}")
    
    for idx, img_info in enumerate(image_data):
        img_path = os.path.join(data_folder, img_info['filename'])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Add frame number and angle overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_text = f"Frame: {img_info['frame_num']}"
        angle_text = f"Angle: {img_info['angle']:.2f}" if img_info['angle'] is not None else ""
        
        # Draw semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Draw text
        cv2.putText(img, frame_text, (20, 50), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        if angle_text:
            cv2.putText(img, angle_text, (20, 85), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        out.write(img)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_data)} frames")
    
    out.release()
    print(f"\nVideo created successfully: {output_video}")
    print(f"\nNext steps:")
    print(f"1. Review the video to identify frame ranges to keep")
    print(f"2. Run the script again with --mode prune and --keep-ranges")
    print(f"   Example: python {os.path.basename(__file__)} --folder {folder_name} --mode prune --keep-ranges '0-100,150-300'")

def parse_ranges(range_str):
    """
    Parse range string like '0-100,150-300,400-500' into list of tuples.
    
    Returns:
        List of (start, end) tuples
    """
    ranges = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            ranges.append((int(start), int(end)))
        else:
            # Single frame
            frame = int(part)
            ranges.append((frame, frame))
    return ranges

def prune_dataset(folder_name, keep_ranges_str):
    """
    Create a pruned dataset by copying only the images in the specified frame ranges.
    
    Args:
        folder_name: Name of the source folder
        keep_ranges_str: String like '0-100,150-300' specifying frames to keep
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    source_folder = os.path.join(script_path, "..", "data", folder_name)
    target_folder = os.path.join(script_path, "..", "data", f"{folder_name}_pruned")
    
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Create target folder
    os.makedirs(target_folder, exist_ok=True)
    
    # Parse ranges
    keep_ranges = parse_ranges(keep_ranges_str)
    print(f"Frame ranges to keep: {keep_ranges}")
    
    # Get all image files
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    
    # Parse images
    image_data = []
    for img_file in image_files:
        frame_num = parse_frame_number(img_file)
        if frame_num is not None:
            image_data.append({
                'filename': img_file,
                'frame_num': frame_num
            })
    
    # Filter images based on ranges
    kept_images = []
    for img_info in image_data:
        frame_num = img_info['frame_num']
        for start, end in keep_ranges:
            if start <= frame_num <= end:
                kept_images.append(img_info)
                break
    
    print(f"Keeping {len(kept_images)} out of {len(image_data)} images")
    
    # Copy images to target folder
    import shutil
    for idx, img_info in enumerate(kept_images):
        source_path = os.path.join(source_folder, img_info['filename'])
        target_path = os.path.join(target_folder, img_info['filename'])
        shutil.copy2(source_path, target_path)
        
        if (idx + 1) % 100 == 0:
            print(f"Copied {idx + 1}/{len(kept_images)} images")
    
    print(f"\nPruned dataset created successfully: {target_folder}")
    print(f"Total images: {len(kept_images)}")

def main():
    parser = argparse.ArgumentParser(
        description='Create video from dataset images and prune dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create review video
  python create_video_and_prune.py --folder train_3 --mode video
  
  # Create pruned dataset keeping frames 0-100, 150-300, and 400-500
  python create_video_and_prune.py --folder train_3 --mode prune --keep-ranges "0-100,150-300,400-500"
  
  # Create pruned dataset keeping only frames 50-200
  python create_video_and_prune.py --folder train_3 --mode prune --keep-ranges "50-200"
        """
    )
    
    parser.add_argument('--folder', type=str, required=True,
                        help='Name of the folder in data/ containing images')
    parser.add_argument('--mode', type=str, choices=['video', 'prune'], required=True,
                        help='Mode: "video" to create review video, "prune" to create pruned dataset')
    parser.add_argument('--keep-ranges', type=str,
                        help='Frame ranges to keep (e.g., "0-100,150-300"). Required for prune mode.')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for video (default: 10)')
    parser.add_argument('--output', type=str,
                        help='Output video path (default: data/{folder}_review.mp4)')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        create_video_with_frame_numbers(args.folder, args.output, args.fps)
    elif args.mode == 'prune':
        if not args.keep_ranges:
            print("Error: --keep-ranges is required for prune mode")
            parser.print_help()
            sys.exit(1)
        prune_dataset(args.folder, args.keep_ranges)

if __name__ == '__main__':
    main()
