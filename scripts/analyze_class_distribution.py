#!/usr/bin/env python
"""
Script to analyze class distribution in a collected dataset and optionally balance it.

Usage:
    # Analyze distribution
    python analyze_class_distribution.py --folder train_new
    
    # Analyze with visualization
    python analyze_class_distribution.py --folder train_new --visualize
    
    # Balance dataset to baseline (300/class)
    python analyze_class_distribution.py --folder train_new --balance --balance-target baseline
    
    # Balance dataset to ideal (500/class)  
    python analyze_class_distribution.py --folder train_new --balance --balance-target ideal
    
    # Balance to minimum class count (auto)
    python analyze_class_distribution.py --folder train_new --balance --balance-target auto
"""

import os
import sys
import argparse
import re
from collections import defaultdict
import numpy as np
import shutil
import random

def parse_angle(filename):
    """Extract angle from filename like '000000-0.50.jpg'"""
    match = re.match(r'\d{6}([-]?\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def classify_angle(angle):
    """
    Classify angle into one of 5 classes:
    0: hard-left (-0.5 to -0.3)
    1: left (-0.3 to -0.1)
    2: straight (-0.1 to 0.1)
    3: right (0.1 to 0.3)
    4: hard-right (0.3 to 0.5)
    """
    if angle <= -0.5:
        return 0 
    elif angle < 0:
        return 1
    elif angle == 0:
        return 2
    elif angle < 0.5:
        return 3
    elif angle == 0.5:
        return 4
    else:
        return None  # Out of range

def analyze_distribution(folder_name, visualize=False):
    """Analyze the class distribution in a dataset folder."""
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
    
    # Count images per class
    class_counts = defaultdict(int)
    angle_list = []
    out_of_range = 0
    
    for img_file in image_files:
        angle = parse_angle(img_file)
        if angle is not None:
            angle_list.append(angle)
            cls = classify_angle(angle)
            if cls is not None:
                class_counts[cls] += 1
            else:
                out_of_range += 1
    
    total_images = len(image_files)
    valid_images = sum(class_counts.values())
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Dataset: {folder_name}")
    print(f"{'='*60}\n")
    
    print(f"Total images: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Out of range: {out_of_range}\n")
    
    class_names = {
        0: "Hard-Left (-0.5 to -0.3)",
        1: "Left (-0.3 to -0.1)",
        2: "Straight (-0.1 to 0.1)",
        3: "Right (0.1 to 0.3)",
        4: "Hard-Right (0.3 to 0.5)"
    }
    
    print(f"{'Class':<8} {'Name':<25} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print(f"{'-'*60}")
    
    for cls in range(5):
        count = class_counts[cls]
        percentage = (count / valid_images * 100) if valid_images > 0 else 0
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        
        print(f"{cls:<8} {class_names[cls]:<25} {count:<10} {percentage:>6.2f}%     {bar}")
    
    # Statistics
    if angle_list:
        print(f"\n{'='*60}")
        print(f"Angle Statistics")
        print(f"{'='*60}\n")
        print(f"Mean angle: {np.mean(angle_list):.4f}")
        print(f"Std dev:    {np.std(angle_list):.4f}")
        print(f"Min angle:  {np.min(angle_list):.4f}")
        print(f"Max angle:  {np.max(angle_list):.4f}")
    
    # Balance assessment
    print(f"\n{'='*60}")
    print(f"Balance Assessment")
    print(f"{'='*60}\n")
    
    if valid_images > 0:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"Min class count: {min_count}")
        print(f"Max class count: {max_count}")
        print(f"Balance ratio:   {balance_ratio:.2f} (1.0 is perfect)")
        
        if balance_ratio >= 0.8:
            print("✓ Dataset is well balanced")
        elif balance_ratio >= 0.6:
            print("⚠ Dataset has minor imbalance")
        else:
            print("✗ Dataset is significantly imbalanced")
        
        # Recommendations
        print(f"\n{'='*60}")
        print(f"Recommendations")
        print(f"{'='*60}\n")
        
        target_baseline = 300
        target_ideal = 500
        
        shortfall_baseline = {cls: max(0, target_baseline - class_counts[cls]) for cls in range(5)}
        shortfall_ideal = {cls: max(0, target_ideal - class_counts[cls]) for cls in range(5)}
        
        print(f"For baseline dataset (300/class):")
        for cls in range(5):
            if shortfall_baseline[cls] > 0:
                print(f"  - Need {shortfall_baseline[cls]} more images for class {cls} ({class_names[cls]})")
            else:
                print(f"  ✓ Class {cls} meets baseline target")
        
        print(f"\nFor ideal dataset (500/class):")
        for cls in range(5):
            if shortfall_ideal[cls] > 0:
                print(f"  - Need {shortfall_ideal[cls]} more images for class {cls} ({class_names[cls]})")
            else:
                print(f"  ✓ Class {cls} meets ideal target")
    
    # Visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart
            classes = list(range(5))
            counts = [class_counts[cls] for cls in classes]
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
            
            ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Class', fontsize=12)
            ax1.set_ylabel('Number of Images', fontsize=12)
            ax1.set_title(f'Class Distribution - {folder_name}', fontsize=14, fontweight='bold')
            ax1.set_xticks(classes)
            ax1.set_xticklabels(['Hard-L', 'Left', 'Straight', 'Right', 'Hard-R'])
            ax1.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax1.text(i, count + 10, str(count), ha='center', fontsize=10, fontweight='bold')
            
            # Histogram of angles
            ax2.hist(angle_list, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Steering Angle', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Angle Distribution', fontsize=14, fontweight='bold')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Center')
            ax2.grid(axis='y', alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(script_path, "..", "data", f"{folder_name}_distribution.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")
            plt.show()
            
        except ImportError:
            print("\nNote: matplotlib not installed. Install with 'pip install matplotlib' for visualizations.")

def balance_dataset(folder_name, target='baseline', output_suffix='_balanced', custom_distribution=None):
    """
    Create a balanced dataset by pruning over-represented classes.
    Ensures each class has at least the baseline target (300 images) before balancing.
    
    Args:
        folder_name: Name of the source folder
        target: 'baseline' (300/class), 'ideal' (500/class), 'auto' (balance to min class), or 'custom'
        output_suffix: Suffix for the output folder name
        custom_distribution: List of 5 integers specifying target count for each class [c0, c1, c2, c3, c4]
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    source_folder = os.path.join(script_path, "..", "data", folder_name)
    target_folder = os.path.join(script_path, "..", "data", f"{folder_name}{output_suffix}")
    
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Get all image files and classify them
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    
    if len(image_files) == 0:
        print(f"Error: No images found in '{source_folder}'")
        return
    
    # Organize images by class
    class_images = {0: [], 1: [], 2: [], 3: [], 4: []}
    out_of_range = []
    
    for img_file in image_files:
        angle = parse_angle(img_file)
        if angle is not None:
            cls = classify_angle(angle)
            if cls is not None:
                class_images[cls].append(img_file)
            else:
                out_of_range.append(img_file)
    
    class_names = {
        0: "Hard-Left",
        1: "Left",
        2: "Straight",
        3: "Right",
        4: "Hard-Right"
    }
    
    # Print current distribution
    print(f"\n{'='*60}")
    print(f"Balancing Dataset: {folder_name}")
    print(f"{'='*60}\n")
    
    print("Current class distribution:")
    for cls in range(5):
        print(f"  Class {cls} ({class_names[cls]:<12}): {len(class_images[cls])} images")
    
    # Determine target count per class
    if target == 'custom' and custom_distribution:
        # Custom distribution: different target for each class
        target_per_class = {cls: custom_distribution[cls] for cls in range(5)}
        print(f"\nUsing custom distribution:")
        for cls in range(5):
            print(f"  Class {cls} ({class_names[cls]:<12}): target = {target_per_class[cls]}")
    elif target == 'auto':
        # Auto: balance to minimum class count
        min_class_count = min(len(class_images[cls]) for cls in range(5))
        target_per_class = {cls: min_class_count for cls in range(5)}
    else:
        # Uniform distribution (baseline or ideal)
        target_counts = {
            'baseline': 300,
            'ideal': 500
        }
        uniform_target = target_counts.get(target, 300)
        target_per_class = {cls: uniform_target for cls in range(5)}
    
    # Check if we can meet the target for each class
    insufficient_classes = []
    for cls in range(5):
        if len(class_images[cls]) < target_per_class[cls]:
            insufficient_classes.append((cls, class_names[cls], len(class_images[cls]), target_per_class[cls]))
    
    if insufficient_classes:
        print(f"\n⚠ WARNING: Cannot meet target for some classes!")
        for cls, name, current, needed in insufficient_classes:
            shortfall = needed - current
            print(f"   Class {cls} ({name}): has {current}, needs {needed} (shortfall: {shortfall})")
        print(f"\nOptions:")
        print(f"  1. Collect more data for under-represented classes")
        print(f"  2. Use --balance-target auto to balance to minimum class count")
        print(f"  3. Adjust --custom-distribution to match available data")
        print(f"  4. Accept imbalanced dataset and use class weights during training")
        return
    
    # Create output folder
    if os.path.exists(target_folder):
        response = input(f"\nFolder '{target_folder}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Balancing cancelled.")
            return
        shutil.rmtree(target_folder)
    
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Balancing Dataset")
    print(f"{'='*60}\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    total_copied = 0
    total_discarded = 0
    
    for cls in range(5):
        images = class_images[cls]
        current_count = len(images)
        target_count = target_per_class[cls]
        
        if current_count > target_count:
            # Randomly select target_count images
            selected = random.sample(images, target_count)
            discarded = current_count - target_count
            total_discarded += discarded
            print(f"Class {cls} ({class_names[cls]:<12}): Keeping {target_count}/{current_count} (discarding {discarded})")
        else:
            # Keep all images
            selected = images
            print(f"Class {cls} ({class_names[cls]:<12}): Keeping all {current_count} images")
        
        # Copy selected images
        for img_file in selected:
            source_path = os.path.join(source_folder, img_file)
            target_path = os.path.join(target_folder, img_file)
            shutil.copy2(source_path, target_path)
            total_copied += 1
    
    
    print(f"\n{'='*60}")
    print(f"Balancing Complete")
    print(f"{'='*60}\n")
    print(f"Total images copied: {total_copied}")
    print(f"Total images discarded: {total_discarded}")
    print(f"\nFinal distribution:")
    for cls in range(5):
        print(f"  Class {cls} ({class_names[cls]:<12}): {target_per_class[cls]} images")
    print(f"Output folder: {target_folder}")
    print(f"\n✓ Balanced dataset created successfully!")
    
    # Calculate balance ratio
    min_target = min(target_per_class.values())
    max_target = max(target_per_class.values())
    balance_ratio = min_target / max_target if max_target > 0 else 0
    print(f"\nBalance ratio: {balance_ratio:.2f} (1.0 is perfect balance)")


def main():

    parser = argparse.ArgumentParser(
        description='Analyze class distribution in collected dataset and optionally balance it',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze distribution
  python analyze_class_distribution.py --folder train_3
  
  # Analyze with visualization
  python analyze_class_distribution.py --folder train_3 --visualize
  
  # Balance dataset to baseline (300/class)
  python analyze_class_distribution.py --folder train_3 --balance --balance-target baseline
  
  # Balance dataset to ideal (500/class)
  python analyze_class_distribution.py --folder train_3 --balance --balance-target ideal
  
  # Balance dataset to minimum class count (auto)
  python analyze_class_distribution.py --folder train_3 --balance --balance-target auto
  
  # Custom distribution (e.g., fewer hard turns, more straight)
  python analyze_class_distribution.py --folder train_3 --balance --balance-target custom \\
      --custom-distribution "200,300,500,300,200"
        """
    )
    
    parser.add_argument('--folder', type=str, required=True,
                        help='Name of the folder in data/ to analyze')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots (requires matplotlib)')
    parser.add_argument('--balance', action='store_true',
                        help='Create a balanced dataset by pruning over-represented classes')
    parser.add_argument('--balance-target', type=str, default='baseline',
                        choices=['baseline', 'ideal', 'auto', 'custom'],
                        help='Target for balancing: baseline (300/class), ideal (500/class), auto (min class count), or custom')
    parser.add_argument('--custom-distribution', type=str,
                        help='Comma-separated list of 5 integers for custom distribution (e.g., "200,300,500,300,200" for classes 0-4)')
    parser.add_argument('--output-suffix', type=str, default='_balanced',
                        help='Suffix for the balanced output folder (default: _balanced)')
    
    args = parser.parse_args()
    
    # Parse custom distribution if provided
    custom_dist = None
    if args.balance_target == 'custom':
        if not args.custom_distribution:
            print("Error: --custom-distribution is required when using --balance-target custom")
            print("Example: --custom-distribution '200,300,500,300,200'")
            parser.print_help()
            sys.exit(1)
        try:
            custom_dist = [int(x.strip()) for x in args.custom_distribution.split(',')]
            if len(custom_dist) != 5:
                raise ValueError("Must provide exactly 5 values")
            if any(x < 0 for x in custom_dist):
                raise ValueError("All values must be non-negative")
        except ValueError as e:
            print(f"Error parsing custom distribution: {e}")
            print("Expected format: '200,300,500,300,200' (5 comma-separated integers)")
            sys.exit(1)
    
    if args.balance:
        balance_dataset(args.folder, args.balance_target, args.output_suffix, custom_dist)
    else:
        analyze_distribution(args.folder, args.visualize)


if __name__ == '__main__':
    main()
