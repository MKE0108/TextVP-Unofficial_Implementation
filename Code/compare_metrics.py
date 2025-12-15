"""
    python Code/compare_metrics.py \
    --dir1 visii/results \
    --name1 Visii \
    --dir2 experiments \
    --name2 TextVP \
    --output metrics_comparison/dog.png     \
    --prefix dog
"""


import os
import json
import argparse
import matplotlib
# 強制使用非互動式後端，確保不彈出視窗，且在伺服器環境下不報錯
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='比較兩個實驗目錄下的 Metrics (僅存檔，不顯示視窗)')
    
    # Baseline 目錄
    parser.add_argument('--dir1', type=str, required=True, help='第一個實驗根目錄 (例如: visii/results)')
    parser.add_argument('--name1', type=str, default='Method 1', help='第一個實驗在圖表上的名稱 (例如: Visii)')

    # Ours 目錄
    parser.add_argument('--dir2', type=str, required=True, help='第二個實驗根目錄 (例如: TextVP/experiments)')
    parser.add_argument('--name2', type=str, default='Method 2', help='第二個實驗在圖表上的名稱 (例如: TextVP)')
    
    parser.add_argument('--output', '-o', type=str, default='comparison_result.png', help='輸出圖片檔名')
    parser.add_argument('--prefix', '-p', type=str, default='', help='只比較名稱以特定字串開頭的資料夾 (例如: "cat")')
    
    parser.add_argument('--metrics', '-m', nargs='+', type=str, default=None, help='指定要比較的指標 (預設: 自動偵測)')

    return parser.parse_args()

def get_metric_from_file(folder_path):
    """
    嘗試讀取 evaluate/evaluation_results.json
    """
    json_path = os.path.join(folder_path, 'evaluate', 'evaluation_results.json')
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('summary', None)
    except Exception as e:
        print(f"讀取錯誤 {json_path}: {e}")
        return None

def load_paired_data(dir1, dir2, prefix=''):
    """ 掃描 dir1 中的子資料夾，並在 dir2 中尋找對應名稱的資料夾 """
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print("錯誤: 輸入的路徑不存在。")
        sys.exit(1)

    # 1. 取得 dir1 下的所有子資料夾名稱
    all_models = [d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))]
    
    # 2. 用前綴過濾出 cat / dog / landscape
    if prefix:
        filtered_models = [m for m in all_models if m.startswith(prefix)]
        print(f"過濾條件 '{prefix}': 找到 {len(filtered_models)} 個資料夾 (總數 {len(all_models)})")
    else:
        filtered_models = all_models
        print(f"未設定過濾條件: 掃描全部 {len(all_models)} 個資料夾")

    data_store = {}
    all_metrics = set()

    for model in filtered_models:
        path1 = os.path.join(dir1, model)
        path2 = os.path.join(dir2, model)

        # 檢查 dir2 是否有同名資料夾
        if not os.path.isdir(path2):
            continue

        res1 = get_metric_from_file(path1)
        res2 = get_metric_from_file(path2)

        if res1 and res2:
            data_store[model] = (res1, res2)
            all_metrics.update(res1.keys())
            all_metrics.update(res2.keys())
        else:
            print(f"略過: {model} (缺少 evaluation_results.json)")

    return data_store, sorted(list(all_metrics))

def plot_comparison(data_store, metrics, name1, name2, output_file):
    if not data_store:
        print("沒有共同的模型數據可供比較。")
        return

    models = sorted(list(data_store.keys()))
    num_metrics = len(metrics)
    
    # 佈局設定
    cols = 2
    rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    # 處理單一指標的情況，確保 axes 也是陣列
    if num_metrics == 1: 
        axes = np.array([axes])
    axes = axes.flatten()

    # 設定長條圖寬度
    x = np.arange(len(models))
    width = 0.35

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        vals1 = [data_store[m][0].get(metric, 0) for m in models]
        vals2 = [data_store[m][1].get(metric, 0) for m in models]

        rects1 = ax.bar(x - width/2, vals1, width, label=name1, color='skyblue', alpha=0.9)
        rects2 = ax.bar(x + width/2, vals2, width, label=name2, color='salmon', alpha=0.9)

        ax.set_title(f'Metric: {metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 數值標註
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)

    # 移除多餘子圖
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"比較圖表已儲存至: {output_file}")
    
    # 釋放記憶體
    plt.close(fig)

if __name__ == "__main__":
    args = parse_arguments()
    
    data, metrics_list = load_paired_data(args.dir1, args.dir2, args.prefix)
    
    if args.metrics:
        metrics_list = [m for m in metrics_list if m in args.metrics]

    plot_comparison(data, metrics_list, args.name1, args.name2, args.output)