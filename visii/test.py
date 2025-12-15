import argparse
import glob
import os
import torch
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
from visii import StableDiffusionVisii

# image_grid 函數用不到了，可以刪除或留著不呼叫

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_number', type=str, default='best')
    parser.add_argument('--log_folder', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--image_folder', type=str, default='./images')

    # 雖然 batch 不用這兩個，但保留以防萬一，設為 1 避免混淆
    parser.add_argument('--number_of_row', type=int, default=1)
    parser.add_argument('--number_of_col', type=int, default=1)

    parser.add_argument('--guidance_scale', type=float, default=7.5) # 建議改 float
    parser.add_argument('--prompt', type=str, default="") # 【修正】解開註解並給預設值
    parser.add_argument('--hybrid_ins', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionVisii.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    image_folder = args.image_folder

    target_image_path = os.path.join(image_folder, '1_0.png')

    # 檢查檔案是否存在
    if os.path.exists(target_image_path):
        list_images = [target_image_path]
    else:
        print(f"Error: Input image not found at {target_image_path}")
        list_images = []

    log_dir = os.path.join(args.log_path, args.log_folder)
    os.makedirs('results', exist_ok=True)

    for img_path in list_images:
        try:
            before_image = Image.open(img_path).convert("RGB").resize((512, 512))
            
            # 固定輸出檔名，供 batch_test 抓取
            save_name = "output_temp.png"
            location = os.path.join('results', save_name)

            checkpoint = os.path.join(log_dir, 'prompt_embeds_{}.pt'.format(args.checkpoint_number))
            if not os.path.exists(checkpoint):
                print(f"Checkpoint not found: {checkpoint}")
                continue
                
            opt_embs = torch.load(checkpoint, map_location="cuda")

            # 移除外層的 for i in range loop，只執行一次
            if args.hybrid_ins:
                with open(os.path.join(log_dir, 'learned_prompt.txt')) as f:
                    init_prompt = f.read()
                
                # 變數名稱統一為 res
                res = pipe.test_concatenate(prompt_embeds=opt_embs,
                    image=before_image,
                    image_guidance_scale=1.5,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=20,
                    prompt=args.prompt,
                    init_prompt=init_prompt,
                    num_images_per_prompt=1,
                    ).images[0]
            else:
                res = pipe.test(prompt_embeds=opt_embs,
                    image=before_image,
                    image_guidance_scale=1.5,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=20,
                    num_images_per_prompt=1,
                    ).images[0]
            
            res.save(location)
            print(f"Saved to {location}")

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")