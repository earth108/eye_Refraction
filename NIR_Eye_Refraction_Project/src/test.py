from graphviz import Digraph


def create_pipeline_flowchart():
    # 初始化图表，设置从上到下的布局 (TB: Top to Bottom)
    dot = Digraph(comment='Preprocessing Pipeline', format='svg')
    dot.attr(rankdir='TB', splines='ortho')  # ortho: 使用折线，更像工程图

    # 全局节点样式
    dot.attr('node', shape='box', style='filled', fillcolor='white',
             fontname='Microsoft YaHei', fontsize='12')

    # --- 1. 输入阶段 ---
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='输入数据 (Input)', style='dashed', color='grey')
        c.node('Start', '原始数据集\n(Raw Dataset)\n包含 UUID 文件夹', shape='cylinder', fillcolor='#E1F5FE')

    # --- 2. 标签解析 (Parser) ---
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='阶段一: 标签解析 (parser.py)', style='rounded', color='#1565C0')
        c.node('ParseName', '文件名解析\n(rsplit 策略)')
        c.node('CheckFormat', '格式校验\n(7个字段?)', shape='diamond', fillcolor='#FFF9C4')
        c.node('AxisNorm', '轴位规范化\n(0°->180°, 取模)')
        c.node('FeatEng', '特征工程\n(生成 sin2A / cos2A)')

    # --- 3. 质量控制 (QC) ---
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='阶段二: 质量控制 (validator.py)', style='rounded', color='#EF6C00')
        c.node('CheckIntegrity', '完整性检查\n(6图齐全?)', shape='diamond', fillcolor='#FFF9C4')
        c.node('ROIMask', '应用垂直 ROI Mask\n(排除头带/设备干扰)')
        c.node('CheckQuality', '曝光/清晰度检查\n(过曝/欠曝/模糊)', shape='diamond', fillcolor='#FFF9C4')
        c.node('CheckIR', '红外反射检测\n(是否闭眼/无光?)', shape='diamond', fillcolor='#FFF9C4')

    # --- 4. ROI 处理 (ROI Processing) ---
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='阶段三: ROI 对齐与裁剪 (process_roi.py)', style='rounded', color='#2E7D32')
        c.node('LocatePupil', '抗干扰瞳孔定位\n(Gaussian + MinMax)')
        c.node('AvgAnchor', '计算平均锚点\n(Average Anchor)\n保证多方向对齐', style='filled,bold',
               fillcolor='#C8E6C9')  # 重点高亮
        c.node('CenterShift', '启用中心偏移?\n(Center Shift)', shape='diamond', fillcolor='#FFF9C4')
        c.node('CropShift', '偏移裁剪框\n(保留边缘纹理)')
        c.node('CropPad', '中心裁剪+补零\n(传统填充)')

    # --- 5. 输出 ---
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='输出产物 (Output)', style='dashed', color='grey')
        c.node('ValidCSV', '清洗后数据集\n(Processed CSV)', shape='note', fillcolor='#C8E6C9')
        c.node('ImgData', '对齐 ROI 图像集\n(224x224 PNG)', shape='folder', fillcolor='#C8E6C9')
        c.node('RejectLog', '异常剔除日志\n(Rejected Logs)', shape='note', fillcolor='#FFCDD2')

    # --- 连接逻辑 ---
    dot.edge('Start', 'ParseName')
    dot.edge('ParseName', 'CheckFormat')

    # 解析失败路径
    dot.edge('CheckFormat', 'RejectLog', label=' 格式错误', color='red')
    dot.edge('CheckFormat', 'AxisNorm', label=' 通过')

    dot.edge('AxisNorm', 'FeatEng')
    dot.edge('FeatEng', 'CheckIntegrity')

    # QC 失败路径
    dot.edge('CheckIntegrity', 'RejectLog', label=' 缺失', color='red')
    dot.edge('CheckIntegrity', 'ROIMask', label=' 通过')

    dot.edge('ROIMask', 'CheckQuality')
    dot.edge('CheckQuality', 'RejectLog', label=' 质量差', color='red')
    dot.edge('CheckQuality', 'CheckIR', label=' 通过')

    dot.edge('CheckIR', 'RejectLog', label=' 无反射', color='red')
    dot.edge('CheckIR', 'LocatePupil', label=' 通过')

    # ROI 逻辑
    dot.edge('LocatePupil', 'AvgAnchor')
    dot.edge('AvgAnchor', 'CenterShift')

    dot.edge('CenterShift', 'CropShift', label=' 是')
    dot.edge('CenterShift', 'CropPad', label=' 否')

    # 汇聚到输出
    dot.edge('CropShift', 'ValidCSV')
    dot.edge('CropPad', 'ValidCSV')
    dot.edge('ValidCSV', 'ImgData')

    # 渲染并保存
    filename = dot.render('pipeline_flowchart', view=True)
    print(f"流程图已生成: {filename}")


if __name__ == '__main__':
    create_pipeline_flowchart()