
# python Code/image_comparison.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 設定圖片資料夾路徑
base_dir = 'image_for_compare'

# 定義列 (Rows) - 對應檔名的前綴
rows = [
    {'key': 'dog', 'label': 'Dog'},
    {'key': 'cat', 'label': 'Cat'},
    {'key': 'mountain', 'label': 'Landscape'}
]

# 定義行 (Columns) - 根據你的需求更新順序與標籤
cols = [
    {'key': '_before.png', 'label': 'Train Source'},      # 原本的 before，改名
    {'key': '_after.png', 'label': 'Train Target'},       # 原本的 after，改名
    {'key': '_before_sample.png', 'label': 'Before'},     # 新增的一欄，在 TextVP 左邊
    {'key': '_TextVP_sample.png', 'label': 'TextVP'},
    {'key': '_visii_sample.png', 'label': 'Visii'}
]

# 建立畫布 (Rows x Columns)
# 因為新增了一欄，現在是 5 行，將 figsize 寬度從 16 加大到 20 以避免太擠
fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(20, 10))
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
            # 若找不到圖片 (例如還沒生成 _before_sample.png)，顯示提示文字
            ax.text(0.5, 0.5, 'Image Not Found', 
                    horizontalalignment='center', verticalalignment='center')
            # 只有在找不到時才印出警告，方便除錯
            print(f"Warning: File not found - {filename}")

        # 移除座標軸
        ax.set_xticks([])
        ax.set_yticks([])

        # 設定上方標題 (只在第一列顯示)
        if r_idx == 0:
            ax.set_title(col_data['label'], fontsize=16, fontweight='bold', pad=20)

        # 設定左側標題 (只在第一行顯示)
        if c_idx == 0:
            ax.set_ylabel(row_data['label'], fontsize=16, fontweight='bold', labelpad=20)

# 自動調整佈局
plt.tight_layout()

# 設定儲存路徑 (存回 image_for_compare 資料夾)
output_filename = 'comparison_image.png'
output_path = os.path.join(base_dir, output_filename)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"比較圖已成功儲存至: {output_path}")
