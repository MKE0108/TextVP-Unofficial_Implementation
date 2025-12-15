# TextVP - Unofficial Implementation

> **Textualize Visual Prompt for Image Editing via Diffusion Bridge** (AAAI'25)  
> ⚠️ 這是非官方的實作版本

## 簡介

本專案是論文 "Textualize Visual Prompt for Image Editing via Diffusion Bridge" 的非官方實現，基於 [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) 框架開發。

![DEMO](./experiments/20251203_014400/cross_replace=[0.2,%201.0],self_replace=0.0,encoded=False,guidance_scale=3.5,/test_image1.png)

## 專案結構

```
├── run_experiment.py        # 訓練/測試腳本 (命令列版本)
├── sample.py               # 推論腳本
├── experiment_config.py     # 實驗配置管理
├── ptp_utils.py            # Prompt-to-Prompt 工具函數
├── seq_aligner.py          # 序列對齊工具
├── image_utils.py          # 圖像處理工具
├── inversion.py            # DDIM Inversion
├── main.ipynb              # 主要訓練/測試 Notebook
├── data_generator/         # 資料生成工具
│   └── prompt2prompt_gen_datapair.ipynb # 生成 Prompt-to-Prompt 資料對 (prompt-to-prompt 版本)
│   └── inp2p.py            # 生成 Prompt-to-Prompt 資料對(intructpix2pix 版本)
├── Dataset                # 資料集
└── experiments/            # 實驗輸出目錄
```

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法一：使用 Notebook

1. 準備資料集放置於 `dataset_old/` 目錄
2. 使用 `main.ipynb` 進行訓練和測試
3. 實驗結果會自動保存至 `experiments/` 目錄

### 方法二：使用命令列腳本 (推薦)

使用 `run_experiment.py` 進行訓練和測試：

#### 訓練

```bash
python run_experiment.py --mode train \
    --exp_dir new \
    --source_image dataset_old/test_1201\(pair_data\)/B_05.png \
    --target_image dataset_old/test_1201\(pair_data\)/A_05.png \
    --test_image_pattern "dataset_old/test_1130\(single_data\)/*.png" \
    --coarse_description "a watercolor painting" \
    --guidance_scale "[1, 3.5, 7.5]" \
    --cross_replace_step "[[0.2, 1.0]]" \
    --self_replace_step "[0.0]" \
    --num_epochs 40 \
    --lr 0.001 \
    --optimizer AdamW
```

#### 測試

```bash
# 測試最新的實驗
python run_experiment.py --mode test

# 測試指定的實驗
python run_experiment.py --mode test --exp_dir 20251203_014400
```

#### 完整流程 (訓練 + 測試)

```bash
python run_experiment.py --mode full \
    --source_image dataset_old/test_1201\(pair_data\)/B_05.png \
    --target_image dataset_old/test_1201\(pair_data\)/A_05.png \
    --test_image_pattern "dataset_old/test_1130(single_data)/*.png" \
    --coarse_description "a watercolor painting" \
    --guidance_scale "[1, 3.5, 7.5]" 
```

#### 訓練參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mode` | (必填) | 模式：`train`、`test` 或 `full` |
| `--exp_dir` | `new` | 實驗目錄名稱（`new` 會自動建立帶時間戳的目錄） |
| `--base_dir` | `experiments/` | 實驗基礎目錄 |
| `--source_image` | (必填) | BEFORE 圖片路徑 |
| `--target_image` | (必填) | AFTER 圖片路徑 |
| `--test_image_pattern` | `""` | 測試圖片的 glob pattern (如 "dataset_old/test_1130(single_data)/*.png" ) |
| `--coarse_description` | `"a watercolor painting"` | 風格描述詞（用於初始化） |
| `--guidance_scale` | `"[7.5]"` | Guidance scale 搜尋列表 |
| `--cross_replace_step` | `"[[0.2, 1.0]]"` | Cross attention 替換範圍 [[begin,end]] |
| `--self_replace_step` | `"[0.0]"` | Self attention 替換範圍 [[begin,end]] ,or [end,..] (預設begin=0)|
| `--encoded_emb` | `"[False]"` | 是否訓練 encoded embedding |
| `--num_epochs` | `50` | 訓練 epoch 數 |
| `--lr` | `0.001` | 學習率 |
| `--optimizer` | `AdamW` | 優化器 (Adam/AdamW/SGD) |
| `--save_interval` | `1` | 每幾個 epoch 儲存一次 |
| `--test_epochs` | `"[0,5,10,20,30,40,49]"` | 測試時使用的 epoch |
| `--device` | `cuda:0` | 使用的設備 |
| `--dtype` | `bfloat16` | 模型資料類型 (`float16`/`bfloat16`/`float32`) |

> **注意**: 列表參數請使用 Python 語法的字串格式，例如 `"[1, 3.5, 7.5]"` 或 `"[[0.2, 1.0]]"`

## 推論 (Sampling)

使用 `sample.py` 對圖片進行風格轉換：

```bash
python sample.py \
    --checkpoint path/to/epoch_X.pt \
    --config path/to/train_config.json \
    --image_dir path/to/input/images \
    --output_dir path/to/output/images
```

### 範例

```bash
python sample.py \
    -c ./experiments/20251203_014400/cross_replace=\[0.2,\ 1.0\],self_replace=0.0,encoded=False,guidance_scale=3.5,/epoch_13.pt \
    -cfg ./experiments/20251203_014400/cross_replace=\[0.2,\ 1.0\],self_replace=0.0,encoded=False,guidance_scale=3.5,/train_config.json \
    -i ./dataset_old_old/test_1130\(single_data\)(single_data) \
    -o ./sample
```

### 參數說明

| 參數 | 縮寫 | 說明 |
|------|------|------|
| `--checkpoint` | `-c` | 訓練好的 `.pt` 檔案路徑 |
| `--config` | `-cfg` | 對應的 `train_config.json` 路徑 |
| `--image_dir` | `-i` | 輸入圖片目錄（或 glob pattern） |
| `--output_dir` | `-o` | 輸出目錄 |
| `--device` | `-d` | 使用的設備（預設 `cuda:0`） |
| `--ext` | `-e` | 圖片副檔名（預設 `png`） |
| `--dtype` | | 模型資料類型（預設 `bfloat16`，可選 `float16`/`float32`） |

## 致謝

- 基於 [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) 開發
- 原論文: [Textualize Visual Prompt for Image Editing via Diffusion Bridge (AAAI'25)](https://arxiv.org/abs/2501.03495)

## Disclaimer

This is an unofficial implementation and is not affiliated with the original authors.

# For Po Han's Update

## For visii
要先另外創一個conda環境 (跟TextVP分開)
並且 ```cd visii```

### 跑training
```
python train.py \
    --image_folder ./Dataset/landscape_sunset \
    --subfolder train \
    --expname landscape_sunset_test  \
    --file_prefix mountain_01   \
    --cuda_device cuda:1
```
- 跑完後結果會存在visii/logs資料夾

### 跑testing
```
python batch_test.py \
    --project_name "landscape_sunset" \
    --script_path "test.py"   \
    --source_dir "Dataset/landscape_sunset/test" \
    --output_dir "results/landscape_sunset_test" \
    --log_folder_name "landscape_sunset" 
```
- 跑完後結果存在visii/results資料夾

## For TextVP

### 資料集解說
例子: Dataset/cat_watercolor
- Before(AI) -> After(AI): 前面5對pair照, 由```data_generator/p2p_gen_datapair.py``` 所生出
- Before(真實圖片) -> After(AI): 後面5對pair照, before找去網路找真實圖片, 接著用data_generator/inp2p.py所生出 
- 第1對pair (AI -> AI) 會放在Dataset/cat_watercolor/train資料夾, 其餘pair都放入Dataset/cat_watercolor/test資料夾
- 也可以用p2p_gen_datapair.py生出before的AI照片, 再用inp2p.py生出after照片 (因為有時p2p_gen_datapair的after效果不好, 結構會跑掉)

### 跑 run_experimnent.py
```
    python Code/run_experiment.py --mode full \
        --device "cuda:1" \
        --exp_dir dog_cat \
        --source_image Dataset/dog_cat/train/dog_01_before.png \
        --target_image Dataset/dog_cat/train/dog_01_after.png \
        --test_image_pattern "Dataset/dog_cat/test/*_before.png" \
        --coarse_description "a cat" \
        --guidance_scale "[1, 3.5, 7.5]" \
        --cross_replace_step "[[0.2, 1.0]]" \
        --num_epochs 50    
```
- 結果會存在experiments/dog_cat裡面

### 跑 sample.py -> 生成圖片
```
    python Code/sample.py \
        --device "cuda:1" \
        --checkpoint experiments/dog_cat/epoch_1.pt \
        --config experiments/dog_cat/train_config.json \
        --image_dir "Dataset/dog_cat/test/*before.png" \
        --output_dir experiments/dog_cat_fail/generated_samples
```
- 結果存在experiments/dog_cat/generated_samples裡面

### 跑 evaluate.py -> 評估指標
for TextVP:
```
    python Code/evaluate.py \
    --input_path experiments/dog_cat/generated_samples \
    --source_path Dataset/dog_cat/test    \
    --output_path experiments/dog_cat/evaluate    \
    --vp_base Dataset/dog_cat/train/dog_01    \
    --device "cuda:1"
```    
- 結果存在experiments/dog_cat/evaluate裡面

for visii:
```
    python Code/evaluate.py \
    --input_path visii/results/cat_watercolor \
    --source_path Dataset/cat_watercolor/test    \
    --output_path visii/results/cat_watercolor/evaluate    \
    --vp_base Dataset/cat_watercolor/train/cat_01    \
    --device "cuda:1"
```
- 結果存在visii/results/cat_watercolor/evaluate裡面


### 做出比較圖:
```
    python Code/compare_metrics.py \
    --dir1 visii/results \
    --name1 Visii \
    --dir2 experiments \
    --name2 TextVP \
    --output metrics_comparison/dog.png     \
    --prefix dog
```
- 結果存在metrics_comparison