"""
    python Code/compare_metrics.py \
    --dir1 visii/results \
    --name1 Visii \
    --dir2 experiments \
    --name2 TextVP \
    --output metrics_comparison/landscape.png \
    --json_output metrics_comparison/landscape_summary.json \
    --prefix landscape
"""

import os
import json
import argparse
import matplotlib
# 強制使用非互動式後端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='比較兩個實驗目錄下的 Metrics 並產出圖表與 JSON 報告')
    
    # Baseline 目錄
    parser.add_argument('--dir1', type=str, required=True, help='第一個實驗根目錄')
    parser.add_argument('--name1', type=str, default='Method 1', help='第一個實驗名稱')

    # Ours 目錄
    parser.add_argument('--dir2', type=str, required=True, help='第二個實驗根目錄')
    parser.add_argument('--name2', type=str, default='Method 2', help='第二個實驗名稱')
    
    # 輸出設定
    parser.add_argument('--output', '-o', type=str, default='comparison_result.png', help='輸出圖片檔名')
    parser.add_argument('--json_output', '-j', type=str, default=None, help='輸出 JSON 摘要檔名 (若未設定，預設為 output 檔名改副檔名)')
    
    parser.add_argument('--prefix', '-p', type=str, default='', help='過濾前綴 (例如: "dog")')
    parser.add_argument('--metrics', '-m', nargs='+', type=str, default=None, help='指定要比較的指標')

    return parser.parse_args()

def get_metric_from_file(folder_path):
    """ 嘗試讀取 evaluate/evaluation_results.json """
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
    """ 掃描並配對資料 """
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print("錯誤: 輸入的路徑不存在。")
        sys.exit(1)

    all_models = [d for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d))]
    
    if prefix:
        filtered_models = [m for m in all_models if m.startswith(prefix)]
        print(f"過濾條件 '{prefix}': 找到 {len(filtered_models)} 個資料夾")
    else:
        filtered_models = all_models
        print(f"掃描全部 {len(all_models)} 個資料夾")

    data_store = {}
    all_metrics = set()

    for model in filtered_models:
        path1 = os.path.join(dir1, model)
        path2 = os.path.join(dir2, model)

        if not os.path.isdir(path2):
            continue

        res1 = get_metric_from_file(path1)
        res2 = get_metric_from_file(path2)

        if res1 and res2:
            data_store[model] = (res1, res2)
            all_metrics.update(res1.keys())
            all_metrics.update(res2.keys())

    return data_store, sorted(list(all_metrics))

def save_summary_json(data_store, metrics, name1, name2, json_path):
    """
    將比對結果整理成 JSON 格式並存檔
    結構包含：個別模型的詳細數據、差異值(Delta)、以及整體平均值
    """
    if not data_store:
        return

    summary = {
        "meta": {
            "method_1": name1,
            "method_2": name2,
            "metrics_analyzed": metrics,
            "count": len(data_store)
        },
        "details": {},
        "average": {}
    }

    # 用來累加數值計算平均
    sums = {m: {name1: 0.0, name2: 0.0, "delta": 0.0} for m in metrics}
    count = len(data_store)

    for model, (res1, res2) in data_store.items():
        model_data = {
            name1: {},
            name2: {},
            "delta": {} # method2 - method1
        }

        for m in metrics:
            val1 = res1.get(m, 0.0)
            val2 = res2.get(m, 0.0)
            diff = val2 - val1

            # 寫入單一模型數據
            model_data[name1][m] = val1
            model_data[name2][m] = val2
            model_data["delta"][m] = diff

            # 累加總合
            sums[m][name1] += val1
            sums[m][name2] += val2
            sums[m]["delta"] += diff

        summary["details"][model] = model_data

    # 計算平均值
    for m in metrics:
        summary["average"][m] = {
            name1: sums[m][name1] / count,
            name2: sums[m][name2] / count,
            "avg_improvement": sums[m]["delta"] / count
        }

    # 確保目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"JSON 摘要報告已儲存至: {json_path}")
    # 順便印出平均結果到 Console，方便查看
    print(f"\n--- {name2} vs {name1} 平均表現 ---")
    for m in metrics:
        avg_diff = summary["average"][m]["avg_improvement"]
        print(f"{m}: {avg_diff:+.4f}")

def plot_comparison(data_store, metrics, name1, name2, output_file):
    if not data_store:
        print("沒有共同的模型數據可供比較。")
        return

    models = sorted(list(data_store.keys()))
    num_metrics = len(metrics)
    
    cols = 2
    rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    if num_metrics == 1: 
        axes = np.array([axes])
    axes = axes.flatten()

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
        # 避免 X 軸標籤重疊，如果太多就旋轉
        rotation = 45 if len(models) > 5 else 0
        ax.set_xticklabels(models, rotation=rotation, ha='right' if rotation else 'center')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # 數值標註 (只在數據量少時顯示，避免擁擠)
        if len(models) <= 10:
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

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"比較圖表已儲存至: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    args = parse_arguments()
    
    data, metrics_list = load_paired_data(args.dir1, args.dir2, args.prefix)
    
    if args.metrics:
        metrics_list = [m for m in metrics_list if m in args.metrics]
    
    # 1. 畫圖
    plot_comparison(data, metrics_list, args.name1, args.name2, args.output)

    # 2. 存 JSON
    # 如果沒有指定 json_output，則將 output 的 .png 換成 .json
    if args.json_output:
        json_path = args.json_output
    else:
        base, _ = os.path.splitext(args.output)
        json_path = base + ".json"
        
    save_summary_json(data, metrics_list, args.name1, args.name2, json_path)