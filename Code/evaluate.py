
""" 
for TextVP:
    python Code/evaluate.py \
    --input_path experiments/dog_cat/generated_samples \
    --source_path Dataset/dog_cat/test    \
    --output_path experiments/dog_cat/evaluate    \
    --vp_base Dataset/dog_cat/train/dog_01    \
    --device "cuda:1"
    
for visii:
    python Code/evaluate.py \
    --input_path visii/results/cat_watercolor \
    --source_path Dataset/cat_watercolor/test    \
    --output_path visii/results/cat_watercolor/evaluate    \
    --vp_base Dataset/cat_watercolor/train/cat_01    \
    --device "cuda:1"
"""
import argparse
import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. 輔助函數：載入模型與特徵提取
# ==========================================

def load_models(device):
    """載入 CLIP 和 LPIPS/PSNR/SSIM 模型"""
    print(f"Loading models on {device}...")
    
    # 載入 CLIP (用於 V-CLIP 和 I-CLIP)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # 載入 LPIPS (用於感知距離)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    # 載入 PSNR 和 SSIM
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    
    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "lpips": lpips_metric,
        "psnr": psnr_metric,
        "ssim": ssim_metric
    }

def get_clip_features(model, processor, image, device):
    """提取圖片的 CLIP Embedding"""
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    # 正規化特徵向量
    return features / features.norm(p=2, dim=-1, keepdim=True)

def load_image_as_tensor(path, size=(512, 512), device="cuda"):
    """讀取圖片並轉為 Tensor (0-1範圍)"""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB").resize(size)
    # 轉為 [C, H, W] 且範圍 [0, 1]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

# ==========================================
# 2. 核心評估邏輯
# ==========================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_path, exist_ok=True)
    
    vp_source_path = f"{args.vp_base}_before.png"
    vp_target_path = f"{args.vp_base}_after.png"
    
    if not os.path.exists(vp_source_path) or not os.path.exists(vp_target_path):
        print(f"Error: Visual Prompt files not found.")
        print(f"Looking for: {vp_source_path} and {vp_target_path}")
        return
    
    # 1. 載入模型
    models = load_models(device)
    
    # 2. 計算 Visual Prompt (參考方向) 的特徵
    # V-CLIP 核心概念：計算 (VP_After - VP_Before) 的向量
    print("Computing Visual Prompt features...")
    vp_source_img = Image.open(vp_source_path).convert("RGB")
    vp_target_img = Image.open(vp_target_path).convert("RGB")
    
    vp_source_feat = get_clip_features(models["clip_model"], models["clip_processor"], vp_source_img, device)
    vp_target_feat = get_clip_features(models["clip_model"], models["clip_processor"], vp_target_img, device)
    
    # 計算參考編輯方向 (Reference Direction)
    vp_direction = vp_target_feat - vp_source_feat
    # 這裡通常不需要再 Normalize vp_direction，因為我們要算的是兩個向量夾角的 Cosine Similarity
    
    # 3. 掃描輸入圖片
    # 假設 input_path 裡面的檔名與 source_path 裡的檔名是一致的
    generated_files = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    
    if not generated_files:
        print(f"Error: No images found in {args.input_path}")
        return

    print(f"Found {len(generated_files)} generated images. Starting evaluation...")
    
    results = {
        "metrics": {},
        "summary": {}
    }
    
    total_scores = {"v_clip": [], "i_clip": [], "psnr": [], "ssim": [], "lpips": []}

    for gen_path in tqdm(generated_files):
        filename = os.path.basename(gen_path)
        
        name_stem = os.path.splitext(filename)[0] # 去掉 .png
        
        if name_stem.endswith("_sample"):
            name_stem = name_stem[:-7]
        # 1. 解析 Clean ID (去掉 _before/_after)
        clean_id = name_stem
        if clean_id.endswith("_before"): 
            clean_id = clean_id[:-7]
        elif clean_id.endswith("_after"): 
            clean_id = clean_id[:-6]
        
        # 尋找對應的原圖 (Source / Before)
        src_path = os.path.join(args.source_path, f"{clean_id}_before.png")
        
        if not os.path.exists(src_path):
            print(f"Warning: Source image not found for {filename}, skipping...")
            continue
            
        # 尋找對應的 Ground Truth (After) - 選用
        gt_path  = os.path.join(args.gt_path, f"{clean_id}_after.png") if args.gt_path else None
        
        if not os.path.exists(src_path):
            print(f"Skip: {clean_id}_before.png not found.") 
            continue

        has_gt = os.path.exists(gt_path)

        # 讀取圖片
        gen_img = Image.open(gen_path).convert("RGB")
        src_img = Image.open(src_path).convert("RGB")
        
        # --- 計算 CLIP 指標 ---
        gen_feat = get_clip_features(models["clip_model"], models["clip_processor"], gen_img, device)
        src_feat = get_clip_features(models["clip_model"], models["clip_processor"], src_img, device)
        
        # 1. I-CLIP (Fidelity): 生成圖與原圖的相似度
        i_clip = torch.nn.functional.cosine_similarity(src_feat, gen_feat).item()
        
        # 2. V-CLIP (Direction): 編輯方向的一致性
        # Test Direction = Gen - Source
        test_direction = gen_feat - src_feat
        v_clip = torch.nn.functional.cosine_similarity(vp_direction, test_direction).item()
        
        scores = {
            "v_clip": v_clip,
            "i_clip": i_clip
        }
        
        # --- 計算 Pixel/Structure 指標 (如果有 GT) ---
        if has_gt:
            # 載入 Tensor 格式
            gen_tensor = load_image_as_tensor(gen_path, device=device)
            gt_tensor = load_image_as_tensor(gt_path, device=device)
            
            # PSNR
            psnr_val = models["psnr"](gen_tensor, gt_tensor).item()
            scores["psnr"] = psnr_val
            
            # SSIM
            ssim_val = models["ssim"](gen_tensor, gt_tensor).item()
            scores["ssim"] = ssim_val
            
            # LPIPS (需要輸入 [-1, 1] 範圍，我們的 tensor 是 [0, 1]，所以要轉換)
            lpips_val = models["lpips"](gen_tensor * 2 - 1, gt_tensor * 2 - 1).item()
            scores["lpips"] = lpips_val
        
        # 紀錄單張分數
        results["metrics"][filename] = scores
        
        # 累加總分
        total_scores["v_clip"].append(v_clip)
        total_scores["i_clip"].append(i_clip)
        if has_gt:
            total_scores["psnr"].append(scores["psnr"])
            total_scores["ssim"].append(scores["ssim"])
            total_scores["lpips"].append(scores["lpips"])

    # 4. 計算平均並存檔
    print("\n" + "="*40)
    print(" Evaluation Summary ")
    print("="*40)
    
    summary = {}
    for metric, values in total_scores.items():
        if values:
            avg_score = np.mean(values)
            summary[metric] = avg_score
            print(f"{metric.upper():<10}: {avg_score:.4f}")
    
    if not total_scores["psnr"]:
        print("(PSNR/SSIM/LPIPS skipped because gt_path was not provided or empty)")

    results["summary"] = summary
    
    output_file = os.path.join(args.output_path, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Editing Results")
    
    # 必要路徑
    parser.add_argument("--input_path", type=str, required=True, help="包含模型生成圖片的資料夾 (Generated Images)")
    parser.add_argument("--source_path", type=str, required=True, help="包含原始測試圖片的資料夾 (Source/Before Images)")
    parser.add_argument("--output_path", type=str, required=True, help="評估報告輸出的資料夾")
    
    # Visual Prompt training 圖
    parser.add_argument("--vp_base", type=str, required=True, help="Visual Prompt 的路徑前綴 (例如 ./assets/ref_image, 程式會自動尋找 ref_image_before.png 和 ref_image_after.png)")
    
    # 選用路徑 (用於計算 PSNR/SSIM/LPIPS)
    parser.add_argument("--gt_path", type=str, default="", help="包含 Ground Truth 圖片的資料夾 (Optional)")
    
    parser.add_argument("--device", type=str, default="cuda:1", help="使用裝置")

    args = parser.parse_args()
    
    # 如果沒給 gt_path, 就預設跟 source_path 一樣
    if not args.gt_path:
        args.gt_path = args.source_path
    main(args)