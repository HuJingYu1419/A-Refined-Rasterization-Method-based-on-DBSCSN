import open3d as o3d
import numpy as np


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

        # 计算点云中心点和坐标范围
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


def load_camera_parameters(cam_params_path):
    try:
        cam_params = o3d.io.read_pinhole_camera_parameters(cam_params_path)
        print("相机参数已加载（保留原始主点）")
        return cam_params  # 直接返回原始参数
    except Exception as e:
        print(f"加载失败，尝试兼容模式: {e}")


def main():
    file_path = r"point_cloud.ply"
    cam_params_path = r"point_cloud.ply.json"

    # 加载相机参数（自动修正主点为整数）
    cam_params = load_camera_parameters(cam_params_path)
    if cam_params is None:
        return

    # 使用相机参数中的窗口尺寸创建窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Load View",
        width=int(cam_params.intrinsic.width),
        height=int(cam_params.intrinsic.height)
    )

    pcd, center = read_point_cloud(file_path)
    if pcd is None:
        return

    # 添加坐标轴和中心点标记
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.translate(center)

    # 添加所有几何体
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.add_geometry(sphere)

    print(f"\n成功加载点云，包含 {len(pcd.points)} 个点")
    print("可视化说明:")
    print("- 红色小球表示点云中心点")
    print("- XYZ坐标轴分别显示为红(X)、绿(Y)、蓝(Z)")

    try:
        view_ctl = vis.get_view_control()
        view_ctl.convert_from_pinhole_camera_parameters(cam_params)
        print("相机参数已应用")

    except Exception as e:
        print(f"应用相机参数失败: {e}")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()