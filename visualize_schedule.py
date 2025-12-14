#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

def parse_schedule_file(filename):
    """解析C++生成的调度表文件"""
    schedule = []
    cycle_per_div = 0
    total_cycles = 0
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {filename}")
    
    # 解析头部信息
    for line in lines:
        if line.startswith('Cycle per div:'):
            cycle_per_div = int(line.split(':')[1].strip())
        elif line.startswith('Total cycles:'):
            total_cycles = int(line.split(':')[1].strip())
        elif 'Schedule Table' in line:
            break
    
    if cycle_per_div == 0:
        raise ValueError("无法解析 'Cycle per div'")
    
    # 解析调度表
    for line in lines:
        line = line.strip()
        if line.startswith('(') and '):' in line:
            try:
                # 解析格式: ( Conv1: (0, 0) ): 0
                layer_part, ord_part = line.rsplit('):', 1)
                layer_name = layer_part.split('(')[1].split(':')[0].strip()
                coords_str = layer_part.split('(')[2].split(')')[0]
                x, y = map(int, coords_str.split(','))
                ord = int(ord_part.strip())
                schedule.append((layer_name, x, y, ord))
            except Exception as e:
                print(f"警告: 无法解析行 '{line}': {e}")
                continue
    
    if not schedule:
        raise ValueError("未找到有效的调度表数据")
    
    return schedule, cycle_per_div, total_cycles

def plot_schedule_gantt(schedule, cycle_per_div, total_cycles, output_file=None):
    """绘制调度甘特图"""
    
    # 按层名分组并排序
    layer_nodes = defaultdict(list)
    for layer_name, x, y, ord in schedule:
        layer_nodes[layer_name].append((x, y, ord))
    
    # 按出现顺序排序层名
    layer_names_ordered = []
    seen = set()
    for layer_name, _, _, _ in schedule:
        if layer_name not in seen:
            layer_names_ordered.append(layer_name)
            seen.add(layer_name)
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 使用颜色映射
    colors = plt.cm.tab20(range(len(layer_names_ordered)))
    layer_colors = dict(zip(layer_names_ordered, colors))
    
    # 绘制矩形
    y_positions = {}
    y_pos = 0
    y_ticks = []
    
    for layer_name in layer_names_ordered:
        nodes = layer_nodes[layer_name]
        y_positions[layer_name] = y_pos
        y_ticks.append(y_pos)
        
        for x, y, ord in nodes:
            start_cycle = ord * cycle_per_div
            width = cycle_per_div
            
            # 创建矩形
            rect = patches.Rectangle(
                (start_cycle, y_pos - 0.4),
                width,
                0.8,
                facecolor=layer_colors[layer_name],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # 添加坐标文本
            text_x = start_cycle + width / 2
            text_y = y_pos
            ax.text(
                text_x,
                text_y,
                f'({x},{y})',
                ha='center',
                va='center',
                fontsize=7,
                color='white' if sum(layer_colors[layer_name][:3]) < 1.5 else 'black',
                fontweight='bold'
            )
        
        y_pos += 1
    
    # 设置图形属性
    ax.set_xlim(0, total_cycles)
    ax.set_ylim(-0.5, len(layer_names_ordered) - 0.5)
    ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convolution Layers', fontsize=12, fontweight='bold')
    ax.set_title('Neural Network Layer Scheduling Gantt Chart', fontsize=14, fontweight='bold')
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(layer_names_ordered, fontsize=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor=layer_colors[name], label=name, alpha=0.8)
        for name in layer_names_ordered
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.15, 1),
        fontsize=8,
        title='Layers',
        title_fontsize=9
    )
    
    plt.tight_layout()
    
    if output_file:
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
            print(f"图表已成功保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("用法: python visualize_schedule.py <调度表文件> [输出图片文件]")
        print("示例: python visualize_schedule.py output.txt schedule.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        schedule, cycle_per_div, total_cycles = parse_schedule_file(input_file)
        print(f"解析成功: {len(schedule)} 个调度节点")
        print(f"Cycle per div: {cycle_per_div}")
        print(f"Total cycles: {total_cycles}")
        print(f"涉及的层数: {len(set(n[0] for n in schedule))}")
        
        plot_schedule_gantt(schedule, cycle_per_div, total_cycles, output_file)
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
