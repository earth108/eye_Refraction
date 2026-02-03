import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SampleLabel:  # FIX: 类名修改为更贴切的名称 (数据容器)
    """定义单样本的结构化标签"""
    sample_id: str
    # 原始标签
    S_R: float  # 右眼球镜
    C_R: float  # 右眼柱镜
    A_R: float  # 右眼轴位 (规范化后 1-180)
    S_L: float  # 左眼球镜
    C_L: float  # 左眼柱镜
    A_L: float  # 左眼轴位 (规范化后 1-180)

    # 派生标签 (用于训练) [cite: 136]
    sin_2A_R: float
    cos_2A_R: float
    sin_2A_L: float
    cos_2A_L: float


def normalize_axis(angle: float) -> float:
    """
    FIX: 轴位规范化逻辑
    将任意角度映射到 1-180 区间
    """
    # 取模处理超出 180 的情况 (例如 185 -> 5)
    angle = angle % 180
    # 处理 0 度情况 (0 -> 180)
    if angle == 0:
        angle = 180
    return angle


def encode_axis(angle_deg: float) -> Tuple[float, float]:
    """
    将轴位角度转换为 sin(2A), cos(2A)
    """
    angle_rad = np.deg2rad(angle_deg)
    return np.sin(2 * angle_rad), np.cos(2 * angle_rad)


def parse_filename(filename: str) -> SampleLabel:
    """
    解析文件名字符串
    """
    # 1. 去除路径和后缀
    base_name = os.path.basename(filename)
    if base_name.endswith(('.png', '.jpg', '.jpeg')):  # 支持更多后缀
        base_name = os.path.splitext(base_name)[0]

    # 2. FIX: 使用 rsplit 从右向左切分 6 次
    # 这样即使 ID 中包含下划线，也不会影响标签的提取
    parts = base_name.rsplit('_', 6)

    if len(parts) != 7:
        raise ValueError(f"文件名格式错误，预期找到 6 个数值标签: {base_name}")

    # 3. 解析字段
    sample_id = parts[0]

    try:
        s_r = float(parts[1])
        c_r = float(parts[2])
        # 左眼
        s_l = float(parts[3])
        c_l = float(parts[4])
        # 轴位 (原始)
        raw_a_r = float(parts[5])
        raw_a_l = float(parts[6])
    except ValueError as e:
        raise ValueError(f"标签数值解析失败: {base_name}") from e

    # 4. FIX: 先规范化轴位，再计算 Sin/Cos
    a_r = normalize_axis(raw_a_r)
    a_l = normalize_axis(raw_a_l)

    sin_r, cos_r = encode_axis(a_r)
    sin_l, cos_l = encode_axis(a_l)

    return SampleLabel(
        sample_id=sample_id,
        S_R=s_r, C_R=c_r, A_R=a_r,
        S_L=s_l, C_L=c_l, A_L=a_l,
        sin_2A_R=sin_r, cos_2A_R=cos_r,
        sin_2A_L=sin_l, cos_2A_L=cos_l
    )


# --- 单元测试 ---
if __name__ == "__main__":
    # 定义 3 个不同类型的正常测试用例
    test_cases = [
        # 用例 1: 标准 UUID 格式
        "00a2a09c-6ebd-4425-8a88-64bc7013a890_-2.75_-0.75_-4_-0.25_175_167",

        # 用例 2: ID 中包含下划线 (验证 rsplit 逻辑)
        "batch_01_patient_05_-1.50_-0.50_-2.00_-0.00_90_45",

        # 用例 3: 轴位边界测试 (验证 0°->180° 和 >180° 的处理)
        "tricky_axis_sample_+0.50_-1.25_+1.00_-2.50_0_190"
    ]

    for test_str in test_cases:
        try:
            result = parse_filename(test_str)
            print(f"✅ 解析成功: {test_str}")
            print("-" * 40)
            print(f"样本 ID : {result.sample_id}")
            print(f"右眼 (R): 球镜={result.S_R:>6}, 柱镜={result.C_R:>6}, 轴位={result.A_R:>5}°")
            print(f"左眼 (L): 球镜={result.S_L:>6}, 柱镜={result.C_L:>6}, 轴位={result.A_L:>5}°")
            print("-" * 40)
            print(f"轴位编码验证 (sin² + cos² Should be 1):")
            print(f"R: {result.sin_2A_R ** 2 + result.cos_2A_R ** 2:.4f}")
            print(f"L: {result.sin_2A_L ** 2 + result.cos_2A_L ** 2:.4f}")
            print("\n")  # 增加换行，区分不同用例

        except Exception as e:
            print(f"❌ 解析失败: {test_str}")
            print(f"   原因: {e}\n")