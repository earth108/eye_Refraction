import sys
import os
import pandas as pd
from tqdm import tqdm
import yaml

# --- 1. 系统路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.io.parser import parse_filename
from src.qc.validator import QualityInspector


def main():
    # --- 2. 加载配置 ---
    config_filename = "preprocess_config.yaml"  # 或者是 test_preprocess_config.yaml
    config_path = os.path.join(project_root, "configs", config_filename)

    if not os.path.exists(config_path):
        print(f"❌ 致命错误: 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 3. 路径设置 ---
    raw_rel = config['paths']['raw_dataset_dir']
    csv_rel = config['paths']['output_csv']
    log_rel = config['paths']['logs_dir']

    RAW_DATA_DIR = os.path.join(project_root, raw_rel)
    OUTPUT_CSV = os.path.join(project_root, csv_rel)
    LOG_FILE = os.path.join(project_root, log_rel, "rejected_samples.csv")

    # --- 4. 初始化 QC ---
    try:
        inspector = QualityInspector(config, project_root=project_root)
    except Exception as e:
        print(f"❌ 初始化 QC 模块失败: {e}")
        return

    # --- 5. 循环处理 ---
    valid_data = []
    rejected_log = []

    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ 错误: 找不到原始数据目录: {RAW_DATA_DIR}")
        return

    folders = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
    print(f"🔍 扫描到 {len(folders)} 个样本，开始清洗...")

    for folder_name in tqdm(folders):
        folder_full_path = os.path.join(RAW_DATA_DIR, folder_name)

        # A. 解析标签
        try:
            label_info = parse_filename(folder_name)
        except Exception as e:
            rejected_log.append(f"{folder_name} | Label Error: {str(e)}")
            continue

        # B. 质量检查
        is_valid, message, valid_images = inspector.check_integrity(folder_full_path)

        if not is_valid:
            rejected_log.append(f"{folder_name} | QC Failed: {message}")
            continue

        # C. 组装数据 (修改部分)
        row = label_info.__dict__.copy()

        # 保存文件夹的相对路径
        # os.path.relpath(目标路径, 基准路径) -> 计算出相对路径
        # 例如: data/raw/uuid_folder
        relative_folder_path = os.path.relpath(folder_full_path, project_root)

        # 统一把反斜杠(\)替换为正斜杠(/)，保证 Linux/Windows 兼容性
        row['folder_path'] = relative_folder_path.replace('\\', '/')

        valid_data.append(row)

    # --- 6. 保存 ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(valid_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n✅ 清洗完成!")
    print(f"   有效样本: {len(df)}")
    print(f"   CSV已保存: {OUTPUT_CSV}")

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(rejected_log))
    print(f"   剔除日志: {LOG_FILE}")


if __name__ == "__main__":
    main()
    # 测试