
""" 
CUDA_VISIBLE_DEVICES=1 python data_generator/inp2p.py \
--input_dir "Dataset/landscape_test/test" --output_dir "Dataset/landscape_test/train" \
--image_guidance 1.2 --guidance_scale 8.5 \
--prompt "Make it a pencil sketch"

default hyperparameter: 
--image_guidance 1.5 --guidance_scale 7.5 \
"""
import argparse
import os
import glob
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Run InstructPix2Pix with arguments")

    parser.add_argument("--input_dir", type=str, required=True, help="輸入圖片的資料夾路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出圖片的資料夾路徑")
    
    parser.add_argument("--prompt", type=str, default="Turn into a minimalist line art", help="編輯指令 (Prompt)")
    parser.add_argument("--gpu_id", type=str, default="1", help="指定 CUDA Device ID")
    parser.add_argument("--steps", type=int, default=50, help="推論步數 (Inference steps)")
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix", help="模型 ID")
    parser.add_argument("--image_guidance", type=float, default=3.0, help="Image Guidance Scale (保留原圖結構的程度)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Text Guidance Scale (遵循文字指令的程度)")
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_id}...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        safety_checker=None
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image_paths = glob.glob(os.path.join(args.input_dir, "*.png")) + \
                  glob.glob(os.path.join(args.input_dir, "*.jpg"))
    
    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images. Processing with prompt: '{args.prompt}'")

    for path in image_paths:
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        # 處理檔名邏輯
        if name.endswith("_before"):
            base_name = name.replace("_before", "")
        else:
            base_name = name

        # 定義 _before 和 _after 的路徑
        save_path_before = os.path.join(args.output_dir, f"{base_name}_before{ext}")
        save_path_after  = os.path.join(args.output_dir, f"{base_name}_after{ext}")

        print(f"Processing: {filename} -> {os.path.basename(save_path_after)}")

        image = Image.open(path).convert("RGB")
        
        # 儲存原始圖為 _before
        image.save(save_path_before)

        # 執行推論
        generated_images = pipe(
            args.prompt, 
            image=image, 
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance,
            guidance_scale=args.guidance_scale
        ).images

        # 儲存生成圖為 _after
        generated_images[0].save(save_path_after)

    print("Done!")

if __name__ == "__main__":
    main()