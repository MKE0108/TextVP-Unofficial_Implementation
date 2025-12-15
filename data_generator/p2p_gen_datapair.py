"""
mountain:
    CUDA_VISIBLE_DEVICES=1 python data_generator/p2p_gen_datapair.py --output_dir Dataset/landscape_winter \
    --file_prefix "mountain"      \
    --prompt_src "A mountain"     \
    --prompt_tgt "A black and white photograph of a mountain."   

dog:
    CUDA_VISIBLE_DEVICES=1 python data_generator/p2p_gen_datapair.py --output_dir Dataset/dog_hat \
    --file_prefix "dog"      \
    --prompt_src "A dog"     \
    --prompt_tgt "A cat"  
    
cat:
    CUDA_VISIBLE_DEVICES=1 python data_generator/p2p_gen_datapair.py --output_dir Dataset/cat_robot \
    --file_prefix "cat"      \
    --prompt_src "A cat"     \
    --prompt_tgt "A dog"  
""" 
import sys
import os
import argparse # 新增 argparse
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from PIL import Image

# 引入 diffusers 相關套件
from diffusers import StableDiffusionPipeline, DDIMScheduler

# 確保可以匯入同目錄下的模組
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 嘗試匯入本地依賴
try:
    import Code.ptp_utils as ptp_utils
    import Code.seq_aligner as seq_aligner
except ImportError:
    print("錯誤: 找不到 ptp_utils.py 或 seq_aligner.py。")
    print("請前往 Google 的 Prompt-to-Prompt GitHub 下載這些輔助檔案。")
    sys.exit(1)

# 全域變數預留 (稍後在 main 中初始化)
device = None
tokenizer = None
ldm_stable = None

# ==========================================
# 參數解析函式
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Prompt-to-Prompt Generation")
    
    parser.add_argument("--output_dir", type=str, default="dataset/p2p_results", help="輸出圖片的資料夾")
    parser.add_argument("--file_prefix", type=str, default="dog", help="輸出檔名的前綴 (例如: dog)")
    
    # Prompt 設定
    parser.add_argument("--prompt_src", type=str, default="a dog", help="原始 Prompt (對應 _before)")
    parser.add_argument("--prompt_tgt", type=str, default="a watercolor painting of a dog", help="目標 Prompt (對應 _after)")
    
    # 生成設定
    parser.add_argument("--num_samples", type=int, default=20, help="要生成幾對圖片")
    parser.add_argument("--steps", type=int, default=50, help="擴散步數")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance Scale")
    parser.add_argument("--gpu_id", type=str, default="1", help="CUDA Device ID")
    parser.add_argument("--seed", type=int, default=-1, help="隨機種子 (設為 -1 則完全隨機)")
    
    # P2P 控制參數 (進階)
    parser.add_argument("--cross_replace_steps", type=float, default=0.8, help="Cross Attention 替換比例")
    parser.add_argument("--self_replace_steps", type=float, default=0.7, help="Self Attention 替換比例")
    
    return parser.parse_args()

# ==========================================
# P2P 核心類別 (維持原樣)
# ==========================================
# 注意: 這些類別會使用全域變數 tokenizer 和 device
MAX_NUM_WORDS = 77

class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

# ==========================================
# 執行函式
# ==========================================

LOW_RESOURCE = False 

def run_and_display(prompts, controller, latent=None, generator=None, num_steps=50, guidance_scale=7.5):
    # 修改為接收參數
    images, x_t = ptp_utils.text2image_ldm_stable(
        ldm_stable, 
        prompts, 
        controller, 
        latent=latent, 
        num_inference_steps=num_steps, 
        guidance_scale=guidance_scale, 
        generator=generator, 
        low_resource=LOW_RESOURCE
    )
    return images, x_t

# ==========================================
# Main Execution
# ==========================================

def main():
    global device, tokenizer, ldm_stable # 宣告使用全域變數
    
    args = parse_args()
    
    # 1. 設定 CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 2. 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 載入模型 (初始化全域變數)
    print(f"Loading Stable Diffusion model on {device}...")
    ldm_stable = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    tokenizer = ldm_stable.tokenizer
    print("Model loaded.")

    # 4. 設定隨機種子
    g_cpu = torch.Generator().manual_seed(args.seed) if args.seed != -1 else None

    print(f"Generating {args.num_samples} pairs...")
    print(f"Source: '{args.prompt_src}'")
    print(f"Target: '{args.prompt_tgt}'")

    for i in range(args.num_samples):
        # 設定 Prompts
        prompts = [args.prompt_src, args.prompt_tgt]

        # 初始化控制器 (使用參數)
        controller = AttentionRefine(
            prompts, 
            args.steps, 
            cross_replace_steps=args.cross_replace_steps,
            self_replace_steps=args.self_replace_steps
        )

        # 執行生成 (每次都用新的 latent，除非指定固定 seed 的邏輯)
        # 注意: 若要每張圖都不一樣，我們不傳入固定的 latent，而是讓它隨機生成
        images, x_t = run_and_display(
            prompts, 
            controller, 
            latent=None, 
            generator=g_cpu,
            num_steps=args.steps,
            guidance_scale=args.guidance_scale
        )

        # 5. 存檔邏輯 (before, after)
        # 檔名格式: {prefix}_{流水號}_before.png
        filename_before = f"{args.file_prefix}_{i:02d}_before.png"
        filename_after  = f"{args.file_prefix}_{i:02d}_after.png"
        
        save_path_before = os.path.join(args.output_dir, filename_before)
        save_path_after  = os.path.join(args.output_dir, filename_after)

        print(f"Pair {i+1}/{args.num_samples}:")
        print(f"  -> {save_path_before}")
        print(f"  -> {save_path_after}")

        # images[0] 是 Source (Before), images[1] 是 Target (After)
        Image.fromarray(images[0]).save(save_path_before)
        Image.fromarray(images[1]).save(save_path_after)

    print("All done!")

if __name__ == "__main__":
    main()