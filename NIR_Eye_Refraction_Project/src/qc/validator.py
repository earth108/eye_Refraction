import os
import cv2
import numpy as np


class QualityInspector:
    def __init__(self, config, project_root=None):
        self.config = config
        self.qc_params = config.get('qc_params', {})
        # [新增] 读取 ROI 参数，为了获取 margin
        self.roi_params = config.get('roi_params', {})

        # 加载阈值
        self.blur_thresh = self.qc_params.get('blur_threshold', 20.0)
        self.dark_thresh = self.qc_params.get('dark_threshold', 3.0)
        self.sat_thresh = self.qc_params.get('sat_threshold', 254.0)
        # 核心阈值：只要眼部区域有亮度超过这个值的点，就认为红外光打进去了
        self.reflection_thresh = self.qc_params.get('reflection_threshold', 150.0)

        # 加载垂直搜索限制 (保持与 ROI 提取逻辑一致)
        # 默认值 0.15 和 0.15 与 process_roi.py 保持同步
        self.margin_top = self.roi_params.get('vertical_margin_top', 0.15)
        self.margin_bottom = self.roi_params.get('vertical_margin_bottom', 0.15)

        print("✅ QC模块初始化完成 (纯 OpenCV 模式 + 垂直区域限制)")

    def check_integrity(self, folder_path):
        """[2.1 文件完整性检查]"""
        valid_images = {}
        # 自动匹配可能的后缀
        valid_exts = ['.png']
        expected_names = [f"es_{i}" for i in range(6)]

        first_shape = None

        for base_name in expected_names:
            found = False
            for ext in valid_exts:
                fname = base_name + ext
                fpath = os.path.join(folder_path, fname)

                if os.path.exists(fpath):
                    # 读取图像
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        return False, f"损坏文件: {fname}", None

                    # 检查尺寸一致性
                    if first_shape is None:
                        first_shape = img.shape
                    elif img.shape != first_shape:
                        return False, f"尺寸不统一: {fname}", None

                    # 单图质检
                    is_good, reason = self.check_image_quality(img)
                    if not is_good:
                        return False, f"质检失败 {fname}: {reason}", None

                    valid_images[fname] = img
                    found = True
                    break

            if not found:
                return False, f"缺失文件: {base_name}", None

        return True, "OK", valid_images

    def check_image_quality(self, image):
        """
        [2.2 图像级质量检查 - OpenCV版]
        改进：只检测中间有效区域，排除顶部和底部的光斑干扰
        """
        h, w = image.shape

        # --- [新增] 定义有效质检区域 (ROI) ---
        y_start = int(h * self.margin_top)
        y_end = int(h * (1 - self.margin_bottom))

        # 防御性编程：防止参数设置错误导致切片为空
        if y_start < y_end:
            # 只取中间部分进行分析
            search_region = image[y_start:y_end, :]
        else:
            # 如果参数有问题，回退到使用全图，但打印警告
            # print(f"⚠️ 警告: Margin 设置不合理 (Top:{y_start} >= Bottom:{y_end})，使用全图质检")
            search_region = image

        # 1. 基础曝光检查 (基于有效区域)
        #    这样如果顶部头带很亮，也不会拉高平均值，使得对眼部欠曝的检查更准确
        mean_val = np.mean(search_region)
        if mean_val < self.dark_thresh:
            return False, f"严重欠曝 (ROI均值:{mean_val:.1f})"
        if mean_val > self.sat_thresh:
            return False, f"严重过曝 (ROI均值:{mean_val:.1f})"

        # 2. 清晰度检查 (基于有效区域)
        #    排除顶部头带纹理对清晰度计算的干扰
        laplacian_var = cv2.Laplacian(search_region, cv2.CV_64F).var()
        if laplacian_var < self.blur_thresh:
            return False, f"模糊 (ROI方差:{laplacian_var:.1f})"

        # 3. 有效性检查 (核心抗干扰逻辑)
        #    只在中间区域寻找最亮光斑。底部设备反光再亮也会被忽略。
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_region)

        if max_val < self.reflection_thresh:
            return False, f"无红外反射斑 (ROI最大亮度:{max_val:.0f} < 阈值)"

        return True, "OK"