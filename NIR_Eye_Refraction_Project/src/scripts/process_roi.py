import sys
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import shutil  # [新增] 用于删除废弃的文件夹

# --- 系统路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


class ROIExtractor:
    def __init__(self, config):
        self.config = config
        self.roi_params = config['roi_params']
        self.target_size = self.roi_params.get('target_size', 224)
        self.max_padding_ratio = self.roi_params.get('max_padding_ratio', 0.4)

        # 填充模式配置
        mode_str = self.roi_params.get('padding_mode', 'constant')
        self.padding_mode = cv2.BORDER_CONSTANT if mode_str == 'constant' else cv2.BORDER_REFLECT
        self.padding_val = self.roi_params.get('padding_value', 0)

        # 搜索区域限制 (去除上下边缘干扰)
        self.margin_top = self.roi_params.get('vertical_margin_top', 0.15)
        self.margin_bottom = self.roi_params.get('vertical_margin_bottom', 0.20)

    def find_pupil_center_robust(self, image_half):
        """
        [抗干扰定位算法]
        只在中间区域搜索最亮点，避开顶部头带和底部设备反光
        """
        h, w = image_half.shape

        # 1. 定义有效搜索区域 (ROI Mask)
        y_start = int(h * self.margin_top)
        y_end = int(h * (1 - self.margin_bottom))

        # 防御性编程：防止 ROI 高度为负
        if y_start >= y_end:
            y_start, y_end = 0, h

        # 截取中间区域进行分析
        search_region = image_half[y_start:y_end, :]

        # 2. 高斯模糊 (去噪)
        blurred = cv2.GaussianBlur(search_region, (7, 7), 0)

        # 3. 寻找最亮点
        _, max_val, _, max_loc_region = cv2.minMaxLoc(blurred)

        # 4. 坐标映射回原图
        # max_loc_region 是相对于 search_region 的 (x, y)
        # 我们需要加上 y_start 偏移量
        center_x = max_loc_region[0]
        center_y = max_loc_region[1] + y_start

        return (center_x, center_y), max_val

    def crop_fixed_size(self, image, center):
        """
        以 center 为中心，剪裁 target_size 大小，使用黑色填充
        """
        cx, cy = center
        half_size = self.target_size // 2

        x1 = cx - half_size
        y1 = cy - half_size
        x2 = cx + half_size
        y2 = cy + half_size

        h, w = image.shape

        # 计算需要填充的量
        pad_top = abs(min(0, y1))
        pad_bottom = max(0, y2 - h)
        pad_left = abs(min(0, x1))
        pad_right = max(0, x2 - w)

        # 有效宽 = 目标宽 - 左填充 - 右填充
        valid_w = max(0, self.target_size - pad_left - pad_right)
        # 有效高 = 目标高 - 上填充 - 下填充
        valid_h = max(0, self.target_size - pad_top - pad_bottom)

        valid_area = valid_w * valid_h
        total_area = self.target_size * self.target_size

        # [优化] 显式转为浮点数计算，防止极少数环境下的整除问题
        padding_ratio = 1.0 - (float(valid_area) / float(total_area))

        # 执行填充
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                self.padding_mode, value=self.padding_val
            )
            # 坐标平移
            x1 += pad_left
            y1 += pad_top
            x2 += pad_left
            y2 += pad_top

        roi = image[y1:y2, x1:x2]
        return roi, padding_ratio


def main():
    # --- 加载配置 ---
    config_filename = "preprocess_config.yaml"
    config_path = os.path.join(project_root, "configs", config_filename)

    if not os.path.exists(config_path):
        print(f"❌ 找不到配置: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 路径
    INPUT_CSV = os.path.join(project_root, config['paths']['output_csv'])
    OUTPUT_DIR = os.path.join(project_root, config['paths']['output_dir'])
    OUTPUT_CSV = os.path.join(project_root, os.path.dirname(config['paths']['output_csv']),
                              "processed_dataset_split.csv")

    extractor = ROIExtractor(config)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入表: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"🚀 开始 ROI 分割处理 (双眼拆分 + 抗干扰), 源样本数: {len(df)}")

    final_rows = []
    padding_reject_log = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['sample_id']
        folder_rel_path = row['folder_path']
        folder_abs_path = os.path.join(project_root, folder_rel_path)

        eyes_info = [
            {'side': 'R', 'suffix': '_R', 'col_prefix': '_R', 'img_part': 'left'},
            {'side': 'L', 'suffix': '_L', 'col_prefix': '_L', 'img_part': 'right'}
        ]

        for eye in eyes_info:
            new_sample_id = f"{sample_id}{eye['suffix']}"
            sample_out_dir = os.path.join(OUTPUT_DIR, new_sample_id)

            # 确保目录存在（如果是重跑，覆盖模式下这个文件夹可能已经有旧图了，不过后续会覆盖）
            os.makedirs(sample_out_dir, exist_ok=True)

            new_row = {
                'sample_id': new_sample_id,
                'original_id': sample_id,
                'side': eye['side'],
                'S': row.get(f"S{eye['col_prefix']}"),
                'C': row.get(f"C{eye['col_prefix']}"),
                'A': row.get(f"A{eye['col_prefix']}"),
                'sin_2A': row.get(f"sin_2A{eye['col_prefix']}"),
                'cos_2A': row.get(f"cos_2A{eye['col_prefix']}"),
            }

            eye_success = True

            for i in range(6):
                img_name = f"es_{i}.png"
                src_path = os.path.join(folder_abs_path, img_name)

                if not os.path.exists(src_path):
                    eye_success = False
                    break

                full_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if full_img is None:
                    eye_success = False
                    break

                h, w = full_img.shape
                mid_x = w // 2

                if eye['img_part'] == 'left':
                    half_img = full_img[:, 0:mid_x]
                else:
                    half_img = full_img[:, mid_x:w]

                try:
                    center, max_val = extractor.find_pupil_center_robust(half_img)
                    roi, padding_ratio = extractor.crop_fixed_size(half_img, center)

                    # --- 质量判定 ---
                    if padding_ratio > extractor.max_padding_ratio:
                        msg = f"{new_sample_id}_es{i}: Padding {padding_ratio:.2%} > {extractor.max_padding_ratio:.0%}"
                        padding_reject_log.append(msg)

                        # 方案 A: 严格剔除
                        eye_success = False
                        break

                    save_name = f"es_{i}.png"
                    save_path = os.path.join(sample_out_dir, save_name)
                    cv2.imwrite(save_path, roi)

                    rel_save_path = os.path.relpath(save_path, project_root).replace('\\', '/')
                    new_row[f'path_{i}'] = rel_save_path

                except Exception as e:
                    print(f"❌ {new_sample_id} - es_{i} 处理错误: {e}")
                    eye_success = False
                    break

            # [重要修正] 后处理：决定是否保留数据
            if eye_success:
                final_rows.append(new_row)
            else:
                # [新增] 如果该样本判定失败（缺图或Padding超标），删除刚刚创建的文件夹
                # 保持数据集目录干净，不留无用文件
                if os.path.exists(sample_out_dir):
                    try:
                        shutil.rmtree(sample_out_dir)
                    except OSError as e:
                        print(f"⚠️ 警告: 无法清理残留目录 {sample_out_dir}: {e}")

    # 保存最终大表
    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    if padding_reject_log:
        log_path = os.path.join(project_root, config['paths']['logs_dir'], "roi_padding_rejects.log")
        with open(log_path, 'w') as f:
            f.write("\n".join(padding_reject_log))
        print(f"⚠️ 已剔除 {len(padding_reject_log)} 组填充过度的样本，日志: {log_path}")

    print(f"\n✅ 处理完成!")
    print(f"   原始样本: {len(df)}")
    print(f"   生成单眼样本: {len(out_df)}")
    print(f"   输出文件: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()