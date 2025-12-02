# TextVP - Unofficial Implementation

> **Textualize Visual Prompt for Image Editing via Diffusion Bridge** (AAAI'25)  
> ⚠️ 這是非官方的實作版本

## 簡介

本專案是論文 "Textualize Visual Prompt for Image Editing via Diffusion Bridge" 的非官方實現，基於 [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) 框架開發。

![DEMO](./experiments/20251203_014400/cross_replace=[0.2,%201.0],self_replace=0.0,encoded=False,guidance_scale=3.5,/test_image1.png)

## 專案結構

```
├── experiment_config.py     # 實驗配置管理
├── ptp_utils.py            # Prompt-to-Prompt 工具函數
├── seq_aligner.py          # 序列對齊工具
├── image_utils.py          # 圖像處理工具
├── inversion.py            # DDIM Inversion
├── main.ipynb              # 主要訓練/測試 Notebook
├── data_generator/         # 資料生成工具
│   └── prompt2prompt_gen_datapair.ipynb # 生成 Prompt-to-Prompt 資料對 (prompt-to-prompt 版本)
│   └── inp2p.py            # 生成 Prompt-to-Prompt 資料對(intructpix2pix 版本)
├── dataset/                # 資料集
└── experiments/            # 實驗輸出目錄
```

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方法

1. 準備資料集放置於 `dataset/` 目錄
2. 使用 `TextVP.ipynb` 進行訓練和測試
3. 實驗結果會自動保存至 `experiments/` 目錄

## 致謝

- 基於 [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) 開發
- 原論文: [Textualize Visual Prompt for Image Editing via Diffusion Bridge (AAAI'25)](https://arxiv.org/abs/2501.03495)

## Disclaimer

This is an unofficial implementation and is not affiliated with the original authors.
