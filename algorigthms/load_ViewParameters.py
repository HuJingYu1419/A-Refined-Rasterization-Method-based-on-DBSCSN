import open3d as o3d
import numpy as np


def load_and_setup_view(point_cloud_path, camera_params_path):
    # 1. 加载点云和相机参数
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    cam_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)

    # 2. 计算相机坐标系信息
    extrinsic = np.array(cam_params.extrinsic).reshape(4, 4)
    R = extrinsic[:3, :3]  # 旋转矩阵
    t = extrinsic[:3, 3]   # 平移向量

    # 相机坐标系各轴方向（世界坐标系下）
    z_axis = R @ np.array([0, 0, 1])  # 相机Z轴方向（视线方向）
    y_axis = R @ np.array([0, 1, 0])  # 相机Y轴方向（上方向）

    # 3. 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=cam_params.intrinsic.width,
                     height=cam_params.intrinsic.height)

    # 4. 添加几何体
    vis.add_geometry(pcd)

    # 添加相机坐标系（红色:X, 绿色:Y, 蓝色:Z）
    cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    cam_coord.transform(extrinsic)
    vis.add_geometry(cam_coord)

    # 5. 设置视角参数
    view_ctl = vis.get_view_control()

    # 计算视角参数
    eye = t                  # 相机位置 = 外参平移量
    lookat = t + z_axis      # 看向Z轴方向前方1米处
    up = y_axis              # 上方向 = 相机Y轴

    # 方法1：直接使用 ViewControl(根据open3d特性有特殊数学处理)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(-up)         #上方向实际为负Y轴
    view_ctl.set_front(-z_axis)  # 视线方向 = 相机负Z轴
    view_ctl.set_zoom(0.01)      # 调整缩放（默认值可能不合适，需手动调整，项目参考0.01）


    # 6. 打印调试信息
    print("=== 相机坐标系参数 ===")
    print(f"相机原点 (eye): {eye}")
    print(f"Z轴方向: {z_axis}")
    print(f"Y轴方向: {y_axis}")
    print(f"视野角度: {view_ctl.get_field_of_view():.2f}°")

    # 7. 运行可视化
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 替换为您的文件路径
    point_cloud_path = "point_cloud.ply"
    camera_params_path = "camera_051.json"

    load_and_setup_view(point_cloud_path, camera_params_path)