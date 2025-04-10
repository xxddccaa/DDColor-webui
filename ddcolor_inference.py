import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import argparse
import shutil
from tqdm import tqdm
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class DDColorModel:
    def __init__(self, model_path, cuda=True):
        """Initialize DDColor colorization model"""
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize modelscope pipeline
        self.model = pipeline(
            Tasks.image_colorization,
            model=model_path,
            device=self.device
        )
        
        print(f"DDColor model loaded successfully from: {model_path}!")
    
    def colorize(self, input_path):
        """Colorize a grayscale image"""
        # Run modelscope pipeline for colorization
        result = self.model(input_path)
        
        # Get the output colorized image (already in BGR format for OpenCV)
        colorized_bgr = result[OutputKeys.OUTPUT_IMG]
        
        return colorized_bgr


def process_images(model, src_dir, dst_dir):
    """Process all images from source directory and save results to destination directory"""
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    
    # Get all files in the source directory
    print(f"Scanning source directory: {src_dir}")
    print(f"Source directory absolute path: {os.path.abspath(src_dir)}")
    print(f"Source directory exists: {os.path.exists(src_dir)}")
    print(f"Destination directory: {dst_dir}")
    print(f"Destination directory absolute path: {os.path.abspath(dst_dir)}")
    print(f"Destination directory exists: {os.path.exists(dst_dir)}")
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist!")
        return
    
    files = os.listdir(src_dir)
    print(f"Found {len(files)} files in source directory")
    # Print some sample filenames
    print(f"Sample files: {', '.join(files[:min(5, len(files))])}")
    
    # Step 1: Copy all real_A files to destination
    real_a_files = [f for f in files if "_real_A" in f]
    print(f"Found {len(real_a_files)} grayscale images with '_real_A' in filename")
    # Print some sample real_A filenames
    if real_a_files:
        print(f"Sample real_A files: {', '.join(real_a_files[:min(5, len(real_a_files))])}")
    
    if len(real_a_files) == 0:
        print("No grayscale images found! Sample files in directory:")
        print(', '.join(files[:min(10, len(files))]))
        return
    
    print("Step 1: Copying all real_A files...")
    for file in tqdm(real_a_files, desc="Copying real_A files"):
        try:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            print(f"DEBUG: Copying {src_path} -> {dst_path}")
            print(f"DEBUG: Source file exists: {os.path.exists(src_path)}")
            
            # Check if source file is readable
            try:
                with open(src_path, 'rb') as f:
                    _ = f.read(10)  # Try to read a few bytes
                print(f"DEBUG: Source file is readable")
            except Exception as e:
                print(f"DEBUG: Source file is not readable: {str(e)}")
            
            shutil.copy2(src_path, dst_path)
            print(f"DEBUG: File copied successfully: {dst_path}")
            print(f"DEBUG: Destination file exists: {os.path.exists(dst_path)}")
        except Exception as e:
            print(f"ERROR copying {file}: {str(e)}")
    
    # Step 2: Copy all real_B_rgb files to destination
    real_b_rgb_files = [f for f in files if "_real_B_rgb" in f]
    print(f"Found {len(real_b_rgb_files)} RGB images with '_real_B_rgb' in filename")
    # Print some sample real_B_rgb filenames
    if real_b_rgb_files:
        print(f"Sample real_B_rgb files: {', '.join(real_b_rgb_files[:min(5, len(real_b_rgb_files))])}")
    
    print("Step 2: Copying all real_B_rgb files...")
    for file in tqdm(real_b_rgb_files, desc="Copying real_B_rgb files"):
        try:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            print(f"DEBUG: Copying {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)
            print(f"DEBUG: File copied successfully: {dst_path}")
        except Exception as e:
            print(f"ERROR copying {file}: {str(e)}")
    
    # Step 3: Process real_A files to generate fake_B_rgb files
    print("Step 3: Processing real_A files to generate fake_B_rgb files...")
    processed_count = 0
    skipped_count = 0
    
    for gray_img in tqdm(real_a_files, desc="Generating colorized images"):
        try:
            # Construct file paths
            gray_path = os.path.join(src_dir, gray_img)
            print(f"\nDEBUG: Processing grayscale image: {gray_path}")
            print(f"DEBUG: Grayscale file exists: {os.path.exists(gray_path)}")
            
            # Check image info
            try:
                img = cv2.imread(gray_path)
                if img is not None:
                    print(f"DEBUG: Image shape: {img.shape}, dtype: {img.dtype}")
                else:
                    print(f"DEBUG: Failed to read image with cv2.imread")
            except Exception as e:
                print(f"DEBUG: Error reading image: {str(e)}")
            
            # Generate fake_B_rgb filename
            fake_rgb_img = gray_img.replace("_real_A", "_fake_B_rgb")
            print(f"DEBUG: Original filename: {gray_img}")
            print(f"DEBUG: New filename: {fake_rgb_img}")
            dst_fake_path = os.path.join(dst_dir, fake_rgb_img)
            print(f"DEBUG: Output path: {dst_fake_path}")
            
            # Colorize the grayscale image
            print(f"DEBUG: Calling colorize method...")
            colorized = model.colorize(gray_path)
            print(f"DEBUG: Colorization successful, result shape: {colorized.shape}")
            
            # Save the colorized result (already in BGR format)
            print(f"DEBUG: Saving to {dst_fake_path}...")
            success = cv2.imwrite(dst_fake_path, colorized)
            print(f"DEBUG: Save successful: {success}")
            print(f"DEBUG: Output file exists: {os.path.exists(dst_fake_path)}")
            
            print(f"Generated colorized image: {os.path.basename(gray_path)} -> {os.path.basename(dst_fake_path)}")
            processed_count += 1
            
        except Exception as e:
            print(f"ERROR processing {gray_img}: {str(e)}")
            import traceback
            print(f"DEBUG: Detailed error traceback:")
            traceback.print_exc()
            skipped_count += 1
    
    print(f"Successfully processed {processed_count} out of {len(real_a_files)} grayscale images")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images due to errors")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DDColor Inference')
    parser.add_argument('--model_path', type=str, default='./DDColormodel',
                      help='Path to the DDColor model directory')
    parser.add_argument('--src_dir', type=str, 
                      default='/ssd/xiedong/image_color/pytorch-CycleGAN-and-pix2pix/results/tongyong_l2ab_4/testA_35/images',
                      help='Source directory containing grayscale and original images')
    parser.add_argument('--dst_dir', type=str, 
                      default='/ssd/xiedong/image_color/ddcolor_test',
                      help='Destination directory for output images')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage instead of GPU')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    print(f"Loading DDColor model from: {args.model_path}")
    print(f"CUDA available: {use_cuda}")
    print(f"Source directory: {args.src_dir}")
    print(f"Destination directory: {args.dst_dir}")
    print(f"Mode: Processing all grayscale images and generating colorized outputs")
    
    # Check if source directory exists
    if not os.path.exists(args.src_dir):
        print(f"Error: Source directory {args.src_dir} does not exist!")
        return
    
    # Load the model
    colorize_model = DDColorModel(args.model_path, use_cuda)
    
    # Process images
    process_images(colorize_model, args.src_dir, args.dst_dir)
    
    print(f"Processing complete. Results saved to {args.dst_dir}")


if __name__ == "__main__":
    main() 