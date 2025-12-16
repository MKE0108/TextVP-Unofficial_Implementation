
# python Code/image_comparison.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 設定圖片資料夾路徑 (請修改為你的實際路徑)
# 假設程式碼跟 image_for_compare 資料夾在同一層
base_dir = 'image_for_compare'

# 定義列 (Rows) - 對應檔名的前綴
# key: 檔名開頭, label: 顯示在圖表左側的名稱
rows = [
    {'key': 'dog', 'label': 'Dog'},
    {'key': 'cat', 'label': 'Cat'},
    {'key': 'mountain', 'label': 'Landscape'}  # 檔名是 mountain，但顯示為 Landscape
]

# 定義行 (Columns) - 對應檔名的後綴
# key: 檔名結尾, label: 顯示在圖表上方的名稱
cols = [
    {'key': '_before.png', 'label': 'Before'},
    {'key': '_after.png', 'label': 'After'},
    {'key': '_TextVP_sample.png', 'label': 'TextVP'},
    {'key': '_visii_sample.png', 'label': 'Visii'}
]

# 建立畫布 (3 列 x 4 行)
fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(16, 10))

# 調整子圖間距
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for r_idx, row_data in enumerate(rows):
    for c_idx, col_data in enumerate(cols):
        ax = axes[r_idx, c_idx]
        
        # 組合完整檔名
        filename = f"{row_data['key']}{col_data['key']}"
        filepath = os.path.join(base_dir, filename)
        
        # 讀取並顯示圖片
        if os.path.exists(filepath):
            img = mpimg.imread(filepath)
            ax.imshow(img)
        else:
            # 如果找不到圖片，顯示文字提示
            ax.text(0.5, 0.5, 'Image Not Found', 
                    horizontalalignment='center', verticalalignment='center')
            print(f"Warning: File not found - {filepath}")

        # 移除座標軸刻度
        ax.set_xticks([])
        ax.set_yticks([])

        # 設定行標題 (只在第一列顯示)
        if r_idx == 0:
            ax.set_title(col_data['label'], fontsize=16, fontweight='bold', pad=20)

        # 設定列標題 (只在第一行顯示，作為該列的 label)
        if c_idx == 0:
            ax.set_ylabel(row_data['label'], fontsize=16, fontweight='bold', labelpad=20)

# 自動調整佈局
plt.tight_layout()

output_filename = 'comparison_image.png'
output_path = os.path.join(base_dir, output_filename)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"比較圖已成功儲存至: {output_path}")
