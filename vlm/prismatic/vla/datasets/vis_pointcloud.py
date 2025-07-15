import numpy as np
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os

def visualize_point_cloud(
    points: Union[np.ndarray, List, Tuple],
    output_path: str = "pointcloud_output",
    output_type: str = "png",
    size: float = 1.0,
    color: Union[str, Tuple[float, float, float]] = "blue",
    background_color: str = "white",
    axis: bool = True,
    title: str = "Point Cloud Visualization",
    views: List[Tuple] = [(0, 0, 1)],  # 默认只保存一个视角
    resolution: Tuple[int, int] = (1920, 1080),
    dpi: int = 300,
    verbose: bool = True
) -> None:
    """
    可视化3D点云并保存为文件（适用于无显示器服务器环境）
    
    参数:
        points: 输入点云，形状为(N,3)的数组
        output_path: 输出文件路径(不带扩展名)
        output_type: 输出类型，支持 'png', 'html', 'ply', 'pcd'
        size: 点的大小(仅对图像输出有效)
        color: 点的颜色，可以是字符串('red','blue'等)或RGB元组(0-1)
        background_color: 背景颜色
        axis: 是否显示坐标轴
        title: 图像标题
        views: 需要保存的视角列表，每个视角是(front_x, front_y, front_z)元组
        resolution: 图像分辨率(宽,高)
        dpi: 图像DPI(仅对png有效)
        verbose: 是否打印处理信息
    
    返回:
        None
    """
    # 转换输入为numpy数组
    points = np.asarray(points)
    assert points.shape[1] == 3, "点云必须是(N,3)的形状"
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if output_type == "png":
        if verbose:
            print(f"正在生成PNG图像到 {output_path}.png...")
        
        fig = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], 
            s=size, 
            c=color if isinstance(color, str) else [color]
        )
        
        ax.set_title(title)
        ax.set_facecolor(background_color)
        fig.set_facecolor(background_color)
        
        if not axis:
            ax.set_axis_off()
        
        plt.savefig(f"{output_path}.png", bbox_inches='tight', dpi=dpi)
        plt.close()
        
    elif output_type == "html":
        if verbose:
            print(f"正在生成交互式HTML到 {output_path}.html...")
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color if isinstance(color, str) else f'rgb{tuple(int(255*c) for c in color)}'
            )
        )])
        
        fig.update_layout(
            scene=dict(bgcolor=background_color),
            title=dict(text=title)
        )
        
        fig.write_html(f"{output_path}.html")
        
    elif output_type in ["ply", "pcd"]:
        if verbose:
            print(f"正在生成{output_type.upper()}文件到 {output_path}.{output_type}...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if output_type == "ply":
            o3d.io.write_point_cloud(f"{output_path}.ply", pcd)
        else:
            o3d.io.write_point_cloud(f"{output_path}.pcd", pcd)
        
    elif output_type == "multi_view":
        if verbose:
            print(f"正在生成多视角PNG图像到 {output_path}_view_[n].png...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=resolution[0], height=resolution[1], visible=False)
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray(
            plt.colors.to_rgb(background_color)
        )
        render_option.point_size = size
        
        for i, view in enumerate(views):
            ctr = vis.get_view_control()
            ctr.set_front(view)
            ctr.set_up((0, 1, 0))  # 设置上方向为Y轴
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f"{output_path}_view_{i}.png")
        
        vis.destroy_window()
    
    else:
        raise ValueError(f"不支持的输出类型: {output_type}。请选择 'png', 'html', 'ply', 'pcd' 或 'multi_view'")
    
    if verbose:
        print("点云可视化完成！")

# 使用示例
if __name__ == "__main__":
    # 生成随机点云示例
    points = np.random.rand(8192, 3)
    
    # 示例1: 保存为PNG
    visualize_point_cloud(points, "output/cloud", "png", 
                        size=0.5, color=(0.1, 0.5, 0.8), 
                        title="Random Point Cloud")
    
    # 示例2: 保存为交互式HTML
    visualize_point_cloud(points, "output/cloud", "html", 
                        size=2.0, color="red")
    
    # 示例3: 保存为PLY文件
    visualize_point_cloud(points, "output/cloud", "ply")
    
    # 示例4: 多视角保存
    visualize_point_cloud(points, "output/cloud_multi", "multi_view",
                         views=[(1,0,0), (0,1,0), (0,0,1), (1,1,1)])