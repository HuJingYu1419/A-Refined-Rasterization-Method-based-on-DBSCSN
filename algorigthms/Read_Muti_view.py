import open3d as o3d
import numpy as np
import os


def read_point_cloud(file_path):
    """通用点云读取函数（支持PLY/PTS等格式）"""
    try:
        if file_path.endswith('.pts'):
            data = np.loadtxt(file_path, skiprows=1, encoding='latin1')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[:, :3])
            if data.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)
        else:
            pcd = o3d.io.read_point_cloud(file_path)

        if not pcd.has_points():
            raise ValueError("点云数据加载失败")

        points = np.asarray(pcd.points)
        center = points.mean(axis=0)
        print("\n点云坐标信息:")
        print(f"X轴范围: {points[:, 0].min():.3f} 到 {points[:, 0].max():.3f}")
        print(f"Y轴范围: {points[:, 1].min():.3f} 到 {points[:, 1].max():.3f}")
        print(f"Z轴范围: {points[:, 2].min():.3f} 到 {points[:, 2].max():.3f}")
        print(f"中心点坐标: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

        return pcd, center
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None


def load_camera_axes(cam_params_path, scale=1.0):
    """加载相机坐标系轴（不应用其他参数）"""
    try:
        cam_params = o3d.io.read_pinhole_camera_parameters(cam_params_path)
        # 创建相机坐标系轴
        cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        # 应用相机外参（仅位置和方向）
        cam_axes.transform(cam_params.extrinsic)
        # 计算相机原点在世界坐标系中的位置
        extrinsic = np.array(cam_params.extrinsic).reshape(4, 4)
        origin = extrinsic[:3, 3]  # 提取平移部分

        print(f"\n已加载相机坐标系轴: {os.path.basename(cam_params_path)}")
        print(f"相机坐标系原点在世界坐标系中的坐标: ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})")
        return cam_axes
    except Exception as e:
        print(f"\n加载相机参数失败（仅显示警告）: {e}")
        return None


def load_all_camera_axes(camera_params_dir, scale=1.0):
    """加载目录中的所有相机参数文件"""
    if not camera_params_dir:
        return []

    if not os.path.isdir(camera_params_dir):
        print(f"\n警告: 指定的相机参数目录不存在: {camera_params_dir}")
        return []

    camera_axes = []
    # 遍历目录中的所有.json文件
    for filename in os.listdir(camera_params_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(camera_params_dir, filename)
            axes = load_camera_axes(filepath, scale=scale)
            if axes:
                camera_axes.append(axes)

    print(f"\n共加载了 {len(camera_axes)} 个相机坐标系")
    return camera_axes


def main():
    # 配置选项
    point_cloud_path = r"point_cloud.ply"
    camera_params_dir = r"D:\DevelopZion\pycharm\store\PointCloud\Trans_Camera_Param"  # 包含多个相机json文件的目录，不需要时填None
    scale = 2  # 坐标轴大小

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer", width=800, height=600)

    # 加载点云
    pcd, center = read_point_cloud(point_cloud_path)
    if pcd is None:
        print("无法加载点云文件")
        vis.destroy_window()
        return

    # 添加点云和世界坐标系
    vis.add_geometry(pcd)
    #world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    #vis.add_geometry(world_axes)

    # 添加中心点标记
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    center_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    center_sphere.translate(center)
    vis.add_geometry(center_sphere)

    # 加载所有相机坐标系
    if camera_params_dir:
        all_cam_axes = load_all_camera_axes(camera_params_dir, scale=scale / 2)
        for cam_axes in all_cam_axes:
            vis.add_geometry(cam_axes)
        print(f"\n共显示了 {len(all_cam_axes)} 个相机坐标系（红色:X, 绿色:Y, 蓝色:Z）")

    print(f"\n成功加载点云，包含 {len(pcd.points)} 个点")
    print("可视化说明:")
    print("- 红色小球表示点云中心点")
    #print("- 世界坐标系轴（黑框）：红(X), 绿(Y), 蓝(Z)")
    if camera_params_dir:
        print(f"- 相机坐标系轴（较小，共 {len(all_cam_axes)} 个）：红(X), 绿(Y), 蓝(Z)")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()