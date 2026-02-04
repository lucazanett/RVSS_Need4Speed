#!/usr/bin/env python
import os
import argparse
import shutil

def get_last_image_number(folder_path):
    """
    Get the last image number from a dataset folder.
    Images are named as: {number:06d}{angle:.2f}.jpg (e.g., 000123-0.50.jpg)
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    
    filenames = os.listdir(folder_path)
    
    if len(filenames) == 0:
        print(f"Warning: Folder '{folder_path}' is empty.")
        return -1
    
    # Extract image numbers from filenames (first 6 characters)
    # Filter only .jpg files
    jpg_files = [f for f in filenames if f.endswith('.jpg')]
    
    if len(jpg_files) == 0:
        print(f"Warning: No .jpg files found in '{folder_path}'.")
        return -1
    
    # Extract the 6-digit number from each filename
    image_numbers = []
    for f in jpg_files:
        try:
            # The first 6 characters are the zero-padded number
            num = int(f[:6])
            image_numbers.append(num)
        except ValueError:
            print(f"Warning: Could not parse number from filename '{f}'")
            continue
    
    if len(image_numbers) == 0:
        print(f"Error: Could not extract valid image numbers from files.")
        return None
    
    # Return the highest number
    return max(image_numbers)


def copy_and_merge_datasets(first_folder, second_folder, joint_folder, last_number_first, dry_run=False):
    """
    Copy images from both datasets into a joint folder with sequential numbering.
    
    Args:
        first_folder: Path to the first dataset folder
        second_folder: Path to the second dataset folder
        joint_folder: Path to the joint output folder
        last_number_first: The last image number from the first dataset
        dry_run: If True, only show what would be copied without actually copying
    """
    # Get files from both datasets
    first_files = sorted([f for f in os.listdir(first_folder) if f.endswith('.jpg')])
    second_files = sorted([f for f in os.listdir(second_folder) if f.endswith('.jpg')])
    
    print(f"\nFirst dataset:  {len(first_files)} images")
    print(f"Second dataset: {len(second_files)} images")
    print(f"Total:          {len(first_files) + len(second_files)} images")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be copied ===")
    else:
        # Create the joint folder if it doesn't exist
        os.makedirs(joint_folder, exist_ok=True)
        print(f"\nCreated/using joint folder: {joint_folder}")
    
    copy_operations = []
    
    # First dataset - copy as is
    print("\n--- First Dataset Files ---")
    for filename in first_files:
        src = os.path.join(first_folder, filename)
        dst = os.path.join(joint_folder, filename)
        copy_operations.append((src, dst, filename))
    
    # Show preview for first dataset
    print(f"Will copy {len(first_files)} files from first dataset (preserving names)")
    if len(first_files) > 0:
        print(f"  First: {first_files[0]}")
        print(f"  Last:  {first_files[-1]}")
    
    # Second dataset - rename to continue numbering
    print("\n--- Second Dataset Files (renumbered) ---")
    current_number = last_number_first + 1
    rename_preview = []
    
    for old_filename in second_files:
        # Extract the angle from the old filename
        try:
            angle_part = old_filename[6:-4]  # Remove .jpg extension and first 6 digits
            new_filename = f"{str(current_number).zfill(6)}{angle_part}.jpg"
            
            src = os.path.join(second_folder, old_filename)
            dst = os.path.join(joint_folder, new_filename)
            copy_operations.append((src, dst, new_filename))
            rename_preview.append((old_filename, new_filename))
            current_number += 1
        except Exception as e:
            print(f"Error processing file '{old_filename}': {e}")
            continue
    
    # Show preview for second dataset
    print(f"Will copy {len(second_files)} files from second dataset (renumbering from {last_number_first + 1})")
    print("\nPreview (first 5 renames):")
    for old, new in rename_preview[:5]:
        print(f"  {old} -> {new}")
    
    if len(rename_preview) > 10:
        print("  ...")
        print("\nPreview (last 5 renames):")
        for old, new in rename_preview[-5:]:
            print(f"  {old} -> {new}")
    
    if dry_run:
        print(f"\nDry run complete. {len(copy_operations)} files would be copied to joint folder.")
        return True
    
    # Actually perform the copies
    print("\nCopying files...")
    success_count = 0
    for src, dst, display_name in copy_operations:
        try:
            shutil.copy2(src, dst)
            success_count += 1
        except Exception as e:
            print(f"Error copying '{display_name}': {e}")
    
    print(f"\nSuccessfully copied {success_count}/{len(copy_operations)} files to joint folder.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Merge two datasets into a joint folder with sequential numbering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview what will be copied
  python augment_ds.py --first_ds train_3 --second_ds train_4 --dry_run
  
  # Actually create the joint dataset
  python augment_ds.py --first_ds train_3 --second_ds train_4
        """
    )
    
    parser.add_argument('--first_ds', type=str, default='train_3',
                        help='First dataset folder name')
    parser.add_argument('--second_ds', type=str, default='train_4',
                        help='Second dataset folder name')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview changes without actually copying files')
    
    args = parser.parse_args()
    
    # Get the script directory and construct data path
    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, "..", "data")
    
    first_ds_path = os.path.join(data_path, args.first_ds)
    second_ds_path = os.path.join(data_path, args.second_ds)
    joint_ds_name = f"joint_dataset_{args.first_ds}_{args.second_ds}"
    joint_ds_path = os.path.join(data_path, joint_ds_name)
    
    print(f"First dataset:  {first_ds_path}")
    print(f"Second dataset: {second_ds_path}")
    print(f"Joint dataset:  {joint_ds_path}")
    
    # Check if joint folder already exists
    if os.path.exists(joint_ds_path) and not args.dry_run:
        print(f"\nWarning: Joint folder '{joint_ds_name}' already exists!")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborting.")
            return
        print("Removing existing joint folder...")
        shutil.rmtree(joint_ds_path)
    
    # Get the last image number from the first dataset
    print(f"\nAnalyzing first dataset '{args.first_ds}'...")
    last_number = get_last_image_number(first_ds_path)
    
    if last_number is None:
        print("Failed to get last image number. Aborting.")
        return
    
    print(f"Last image number in '{args.first_ds}': {last_number}")
    
    # Copy and merge datasets
    print(f"\nMerging datasets into '{joint_ds_name}'...")
    copy_and_merge_datasets(first_ds_path, second_ds_path, joint_ds_path, last_number, dry_run=args.dry_run)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
