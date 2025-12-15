
"""
python batch_test.py \
    --project_name "landscape_sunset" \
    --script_path "test.py"   \
    --source_dir "Dataset/landscape_sunset/test" \
    --output_dir "results/landscape_sunset_test" \
    --log_folder_name "landscape_sunset" 
"""

import os
import shutil
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images for VISII testing")
    
    parser.add_argument("--project_name", type=str, required=True, help="Project name (subfolder in ./images/)")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save final results")
    parser.add_argument("--log_folder_name", type=str, required=True, help="The folder name inside ./logs/ where model is saved")
    
    parser.add_argument("--visii_base_dir", type=str, default="./images")
    parser.add_argument("--script_path", type=str, default="test.py")
    parser.add_argument("--checkpoint_num", type=str, default="best", help="Checkpoint number (e.g., 1000)")
    # 【修正】這裡必須對應 test.py 輸出的檔名 (results/output_temp.png)
    parser.add_argument("--generated_result_path", type=str, default="./results/output_temp.png")
    
    return parser.parse_args()

def main():
    args = parse_args()

    visii_image_dir = os.path.join(args.visii_base_dir, args.project_name)
    
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist.")
        return # 直接結束

    os.makedirs(visii_image_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(args.source_dir) if f.endswith('_before.png')]
    total_images = len(test_images)

    print(f"Found {total_images} images in {args.source_dir}")
    print(f"Project: {args.project_name}")
    print(f"Output: {args.output_dir}")
    print("-" * 30)

    for idx, img_name in enumerate(test_images):
        src_path = os.path.join(args.source_dir, img_name)
        
        # VISII 尋找的目標檔名
        visii_input_path = os.path.join(visii_image_dir, "1_0.png") 
        
        print(f"[{idx+1}/{total_images}] Processing: {img_name}")
        
        shutil.copy(src_path, visii_input_path)
        
        cmd = [
            "python", args.script_path,
            "--image_folder", visii_image_dir, 
            "--log_folder", args.log_folder_name,
            "--number_of_row", "1",
            "--number_of_col", "1",
        ]
        
        try:
            subprocess.run(cmd, check=True) 
        except subprocess.CalledProcessError as e:
            print(f"Error running test.py for image {img_name}: {e}")
            continue

        if os.path.exists(args.generated_result_path):
            output_filename = img_name.replace("_before.png", "_sample.png")
            final_dst_path = os.path.join(args.output_dir, output_filename)
            # 使用 move 覆蓋舊檔
            if os.path.exists(final_dst_path):
                os.remove(final_dst_path)
            shutil.move(args.generated_result_path, final_dst_path)
        else:
            print(f"Warning: Result file not found at {args.generated_result_path}")

    print("-" * 30)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()