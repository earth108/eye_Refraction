import sys
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

# --- 系统路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)


class ROIExtractor:
    def __init__(self, config):
        # ==========================================
        # [控制开关] 在这里修改 True / False
        # False (默认): 保持瞳孔居中，边界补0 (推荐用于回归精度)
        # True  (实验): 移动剪裁框以避免补0，瞳孔将不再居中
        self.ENABLE_CENTER_SHIFT = True
        # ==========================================

        self.config = config
        self.roi_params = config['roi_params']
        self.target_size = self.roi_params.get('target_size', 224)

        # 填充比例阈值
        self.max_padding_ratio = self.roi_params.get('max_padding_ratio', 0.4)

        # 填充模式配置
        mode_str = self.roi_params.get('padding_mode', 'constant')
        self.padding_mode = cv2.BORDER_CONSTANT if mode_str == 'constant' else cv2.BORDER_REFLECT
        self.padding_val = self.roi_params.get('padding_value', 0)

        # 搜索区域限制
        self.margin_top = self.roi_params.get('vertical_margin_top', 0.15)
        self.margin_bottom = self.roi_params.get('vertical_margin_bottom', 0.20)

        if self.ENABLE_CENTER_SHIFT:
            print("⚠️ [警告] 已启用中心偏移策略 (ENABLE_CENTER_SHIFT=True)。")
            print("   剪裁框将偏离瞳孔中心以避免黑边。请确保模型能适应位置变化。")

    def find_pupil_center_robust(self, image_half):
        """
        [抗干扰定位] 返回 (x, y) 和 最大亮度值
        """
        h, w = image_half.shape

        # 1. 定义中间搜索带
        y_start = int(h * self.margin_top)
        y_end = int(h * (1 - self.margin_bottom))

        if y_start >= y_end:
            y_start, y_end = 0, h

        search_region = image_half[y_start:y_end, :]

        # 2. 高斯模糊去噪
        blurred = cv2.GaussianBlur(search_region, (7, 7), 0)

        # 3. 寻找最亮点
        _, max_val, _, max_loc_region = cv2.minMaxLoc(blurred)

        # 4. 坐标还原
        center_x = max_loc_region[0]
        center_y = max_loc_region[1] + y_start

        return (center_x, center_y), max_val

    def crop_fixed_size(self, image, center):
        """
        [核心剪裁逻辑] 支持 补0模式 和 偏移模式
        """
        cx, cy = center
        h, w = image.shape
        half_size = self.target_size // 2

        # --- [新逻辑] 中心偏移策略 ---
        if self.ENABLE_CENTER_SHIFT:
            # 限制中心点的范围，使其不靠边
            # 最小允许中心 = half_size
            # 最大允许中心 = 总长 - half_size
            # 只有当图片尺寸 > target_size 时才生效
            if w > self.target_size:
                cx = max(half_size, min(w - half_size, cx))
            if h > self.target_size:
                cy = max(half_size, min(h - half_size, cy))

        # --- 计算剪裁框坐标 ---
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = int(cx + half_size)
        y2 = int(cy + half_size)

        # --- 计算Padding量 ---
        # 如果启用了 OFFSET 且图片够大，这里的 pad 应该都为 0
        pad_top = abs(min(0, y1))
        pad_bottom = max(0, y2 - h)
        pad_left = abs(min(0, x1))
        pad_right = max(0, x2 - w)

        # --- 计算几何填充比例 (用于质检) ---
        valid_w = max(0, self.target_size - pad_left - pad_right)
        valid_h = max(0, self.target_size - pad_top - pad_bottom)
        valid_area = valid_w * valid_h
        total_area = self.target_size * self.target_size
        padding_ratio = 1.0 - (float(valid_area) / float(total_area))

        # --- 执行填充 ---
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                self.padding_mode, value=self.padding_val
            )
            # 坐标平移 (因为原图变大了，框也要跟着移)
            x1 += pad_left
            y1 += pad_top
            x2 += pad_left
            y2 += pad_top

        # --- 最终剪裁 ---
        roi = image[y1:y2, x1:x2]
        return roi, padding_ratio


def main():
    # --- 配置加载 ---
    # 确保文件名正确
    config_filename = "test_preprocess_config.yaml"
    config_path = os.path.join(project_root, "configs", config_filename)

    if not os.path.exists(config_path):
        print(f"❌ 找不到配置: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 路径定义
    INPUT_CSV = os.path.join(project_root, config['paths']['output_csv'])
    OUTPUT_DIR = os.path.join(project_root, config['paths']['output_dir'])
    OUTPUT_CSV = os.path.join(project_root, os.path.dirname(config['paths']['output_csv']),
                              "processed_dataset_split.csv")
    REJECT_CSV = os.path.join(project_root, config['paths']['logs_dir'], "roi_reject_details.csv")

    extractor = ROIExtractor(config)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入表: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"🚀 开始 ROI 处理")
    print(f"   模式: {'[中心偏移]' if extractor.ENABLE_CENTER_SHIFT else '[零填充补全]'}")
    print(f"   锚点: 6图统一平均锚点")
    print(f"   源样本数: {len(df)}")

    final_rows = []
    reject_records = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['sample_id']
        folder_rel_path = row['folder_path']
        folder_abs_path = os.path.join(project_root, folder_rel_path)

        # 定义左右眼处理元数据
        eyes_info = [
            {'side': 'R', 'suffix': '_R', 'col_prefix': '_R', 'img_part': 'left'},
            {'side': 'L', 'suffix': '_L', 'col_prefix': '_L', 'img_part': 'right'}
        ]

        for eye in eyes_info:
            new_sample_id = f"{sample_id}{eye['suffix']}"
            sample_out_dir = os.path.join(OUTPUT_DIR, new_sample_id)
            os.makedirs(sample_out_dir, exist_ok=True)

            # 准备新行数据
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

            # --- 阶段 1: 收集 6 张图并计算统一锚点 ---
            images_cache = []
            detected_centers = []
            load_success = True

            for i in range(6):
                img_name = f"es_{i}.png"
                src_path = os.path.join(folder_abs_path, img_name)

                if not os.path.exists(src_path):
                    reject_records.append({"Sample_ID": new_sample_id, "Reason": f"缺失文件: {img_name}"})
                    load_success = False
                    break

                # 强制灰度读取
                full_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if full_img is None:
                    reject_records.append({"Sample_ID": new_sample_id, "Reason": f"无法读取: {img_name}"})
                    load_success = False
                    break

                # 切割单眼
                h, w = full_img.shape
                mid_x = w // 2
                if eye['img_part'] == 'left':
                    half_img = full_img[:, 0:mid_x]
                else:
                    half_img = full_img[:, mid_x:w]

                images_cache.append(half_img)

                try:
                    center, max_val = extractor.find_pupil_center_robust(half_img)
                    detected_centers.append(center)
                except Exception:
                    pass

            if not load_success:
                continue

                # --- 阶段 2: 计算平均锚点 (Average Anchor) ---
            if len(detected_centers) == 0:
                reject_records.append({"Sample_ID": new_sample_id, "Reason": "无法定位瞳孔"})
                continue

            avg_x = np.mean([c[0] for c in detected_centers])
            avg_y = np.mean([c[1] for c in detected_centers])
            anchor_center = (int(avg_x), int(avg_y))

            # --- 阶段 3: 统一剪裁并保存 ---
            eye_process_success = True

            for i, img_data in enumerate(images_cache):
                try:
                    # 调用支持 Shift 的剪裁函数
                    roi, padding_ratio = extractor.crop_fixed_size(img_data, anchor_center)

                    # 保存为单通道灰度
                    save_name = f"es_{i}.png"
                    save_path = os.path.join(sample_out_dir, save_name)
                    cv2.imwrite(save_path, roi)

                    rel_save_path = os.path.relpath(save_path, project_root).replace('\\', '/')
                    new_row[f'path_{i}'] = rel_save_path

                    # 质检 (如果开启了Shift，padding_ratio通常为0，除非图太小)
                    if padding_ratio > extractor.max_padding_ratio:
                        reason = f"es_{i} 填充比例 {padding_ratio:.1%} > {extractor.max_padding_ratio:.0%}"
                        reject_records.append({"Sample_ID": new_sample_id, "Reason": reason})
                        eye_process_success = False
                        break

                except Exception as e:
                    reject_records.append({"Sample_ID": new_sample_id, "Reason": f"剪裁异常: {e}"})
                    eye_process_success = False
                    break

            if eye_process_success:
                final_rows.append(new_row)

    # --- 结果保存 ---
    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    if reject_records:
        reject_df = pd.DataFrame(reject_records)
        reject_df = reject_df[["Sample_ID", "Reason"]]
        reject_df.to_csv(REJECT_CSV, index=False, encoding='utf-8-sig')
        print(f"⚠️ 已剔除 {len(reject_records)} 组样本，详见: {REJECT_CSV}")
    else:
        print("✅ 无样本剔除")

    print(f"\n✅ 处理完成! 输出: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()