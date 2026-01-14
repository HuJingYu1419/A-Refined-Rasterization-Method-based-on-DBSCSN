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
        return pcd
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def save_view(vis, file_path):
    """保存相机参数的回调函数"""
    try:
        name = file_path
        filename = f"{name}.json"
        view_ctl = vis.get_view_control()
        cam_params = view_ctl.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, cam_params)
        print(f"相机参数已保存到 {filename}")
        return False
    except Exception as e:
        print(f"保存失败: {e}")
        return False


def main():

    file_path = r"point_cloud.ply" #导入点云文件
    pcd = read_point_cloud(file_path)

    #定义相机分辨率
    """
    分辨率参考：
    Open3d默认：800×600
    720P：1280×720
    1080P：1920×1080
    2K：2560×1440
    4K：3840×2160
    """
    width=800
    height=600
    print(f"当前选择的分辨率为{width}*{height}")
    print("注意请以打开时的窗口大小关闭窗口，否则会保存错误的分辨率")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Capture View", width=width, height=height)

    if pcd is None:
        vis.destroy_window()
        return

    vis.add_geometry(pcd)
    vis.run()
    save_view(vis, file_path)
    vis.destroy_window()


if __name__ == "__main__":
    main()