import open3d as o3d
import numpy as np
import time

from sympy import false


def load_point_cloud(file_path):
    """加载点云文件"""
    print(f"\n[1/4] 加载点云文件: {file_path}")
    start_time = time.time()

    if file_path.endswith('.pts'):
        data = np.loadtxt(file_path, skiprows=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)
    else:
        pcd = o3d.io.read_point_cloud(file_path)

    if not pcd.has_points():
        raise ValueError("点云数据加载失败")

    print(f"原始点云包含 {len(pcd.points)} 个点")
    print(f"加载耗时: {time.time() - start_time:.2f}秒")
    return pcd


def downsample_point_cloud(pcd, method='voxel', scale=0.05):
    """
    降采样点云
    :param method: 'voxel'体素降采样 | 'uniform'均匀降采样
    :param scale: 体素大小(m)或采样间隔(点数量)
    """
    print(f"\n[2/4] 开始降采样 ({method}方法)...")
    start_time = time.time()

    if method == 'voxel':
        down_pcd = pcd.voxel_down_sample(voxel_size=scale)
        print(f"体素大小: {scale}m")
    elif method == 'uniform':
        down_pcd = pcd.uniform_down_sample(every_k_points=int(scale))
        print(f"采样间隔: 每{scale}个点取1个")
    else:
        raise ValueError("方法必须是'voxel'或'uniform'")

    print(f"降采样后点数: {len(down_pcd.points)}")
    print(f"降采样耗时: {time.time() - start_time:.2f}秒")
    return down_pcd


def visualize_point_cloud(pcd):
    """可视化单个点云"""
    print("\n[3/4] 可视化点云...")

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="预处理点云可视化")

    # 添加几何体
    vis.add_geometry(pcd)

    ''''# 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深色背景

    # 添加坐标系（可选）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axis)'''

    print("可视化窗口已打开（关闭窗口继续执行）...")
    vis.run()
    vis.destroy_window()


def save_point_cloud(pcd, output_path):
    """保存点云文件"""
    print(f"\n[4/4] 保存点云到 {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)
    print("保存完成！")


def main():
    # 参数配置
    input_path = "D:\DevelopZion\pycharm\store\PointCloud\setup_33\setup_33.pts"  #输入要处理的点云
    name=input_path[-12:-4]
    output_path = f"Process_{name}.ply"
    downsampling_method = 'voxel'  # 'voxel'或'uniform'
    downsampling_scale = 0.03  # 体素大小(m)或采样间隔
    if_downsample=False  #是否降采样（True或False）

    #处理流程
    original_pcd = load_point_cloud(input_path)
    if if_downsample:
        print("\n已采用降采样")
        down_pcd = downsample_point_cloud(original_pcd, downsampling_method, downsampling_scale)
        visualize_point_cloud(down_pcd)  # 可视化降采样结果
        save_point_cloud(down_pcd, output_path)
    else:
        print("\n未采用降采样")
        visualize_point_cloud(original_pcd)  # 可视化原数据
        save_point_cloud(original_pcd, output_path)



if __name__ == "__main__":
    main()