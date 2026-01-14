import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, distance_transform_edt
from tqdm import tqdm
import cv2
import os
import argparse


class PointCloudRasterizer:
    def __init__(self, point_cloud_path, camera_params_dir=None, single_camera_params_path=None):
        """
        初始化栅格化器
        :param point_cloud_path: 点云文件路径
        :param camera_params_dir: 相机参数目录路径（批处理模式）
        :param single_camera_params_path: 单个相机参数文件路径（单文件模式）
        """
        self.pcd = self.load_point_cloud(point_cloud_path)
        self.camera_params_dir = camera_params_dir
        self.single_camera_params_path = single_camera_params_path

        # These will be set when processing each camera
        self.width = None
        self.height = None
        self.intrinsic = None
        self.extrinsic = None
        self.current_camera_name = None

    def load_point_cloud(self, file_path):
        """
        加载点云文件，支持PLY/PTS格式
        """
        print(f"加载点云文件: {file_path}")
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
        print(f"点云包含 {len(pcd.points)} 个点")
        return pcd

    def load_camera_parameters(self, camera_params_path):
        """加载单个相机的参数"""
        self.camera_params = o3d.io.read_pinhole_camera_parameters(camera_params_path)
        self.width = self.camera_params.intrinsic.width
        self.height = self.camera_params.intrinsic.height
        self.intrinsic = self.camera_params.intrinsic.intrinsic_matrix
        self.extrinsic = self.camera_params.extrinsic
        self.current_camera_name = os.path.splitext(os.path.basename(camera_params_path))[0]

    def project_points_to_image(self,R_kind):
        # 原始点云坐标
        points = np.asarray(self.pcd.points)
        print(f"\n处理相机: {self.current_camera_name}")
        print("原始点云:", len(points))

        # 世界坐标系 → 相机坐标系
        if R_kind == "A":
            points_cam = np.dot(self.extrinsic[:3, :3], points.T).T + self.extrinsic[:3, 3]

        elif R_kind == "B":
            points_cam = (points - self.extrinsic[:3, 3]) @ self.extrinsic[:3,
                                                            :3]  # (本质：points_cam = np.dot(self.extrinsic[:3, :3].T, (points - self.extrinsic[:3, 3]).T).T)

        else:
            print("R_kind参数输入错误")

        # === 新增：过滤负深度点 ===
        valid_depth = points_cam[:, 2] > 0  # 仅保留Z>0的点
        points_cam = points_cam[valid_depth]
        points = points[valid_depth]  # 同步过滤原始点云（保证后续in_view数组长度正确不报错）
        print("过滤负Z后:", len(points_cam))

        # 相机坐标系 → 图像平面
        points_img = np.dot(self.intrinsic, points_cam.T).T
        points_img = points_img[:, :2] / points_img[:, 2, np.newaxis]

        # 计算深度值（相机坐标系Z值）
        depths = points_cam[:, 2]

        # 筛选图像范围内的点
        in_view = (points_img[:, 0] >= 0) & (points_img[:, 0] < self.width) & \
                  (points_img[:, 1] >= 0) & (points_img[:, 1] < self.height)
        print("图像范围内:", sum(in_view))

        return points_img[in_view], depths[in_view], points[in_view]

    def fill_nan_with_nearest_neighbor(self, depth_map, max_distance=1, use_morphology=True, edge_width=0):
        """
        使用最近邻插值填充NaN区域，结合闭运算预处理和边缘保护
        :param depth_map: 原始深度图（含NaN）
        :param max_distance: 最大填充距离（像素）
        :param use_morphology: 是否启用闭运算预处理
        :param edge_width: 边缘保护宽度（像素），根据实际需要决定是否启用及其使用距离
        :return: 填充后的深度图
        """
        # 初始化输出
        filled_map = depth_map.copy()

        # 步骤1：创建原始掩膜（有效=255，NaN=0）
        mask = np.where(np.isnan(depth_map), 0, 255).astype(np.uint8)

        # 步骤2：边缘保护（强制边缘NaN不参与填充）
        if edge_width > 0:
            edge_mask = np.zeros_like(mask)
            edge_mask[edge_width:-edge_width, edge_width:-edge_width] = 255
            mask = cv2.bitwise_and(mask, edge_mask)

        # 步骤3：闭运算预处理（仅生成建议填充区，不修改原始mask）
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            new_fill_regions = (closed_mask == 255) & (mask == 0)  # 闭运算新增的填充区
        else:
            new_fill_regions = np.zeros_like(mask, dtype=bool)

        # 步骤4：计算最近邻距离（基于原始mask）
        distances, indices = distance_transform_edt(
            ~(mask > 0),  # 反转：NaN=1，有效=0
            return_distances=True,
            return_indices=True
        )

        # 步骤5：优先填充闭运算建议的小孔洞
        rows_new, cols_new = np.where(new_fill_regions & (distances <= max_distance))
        for r, c in zip(rows_new, cols_new):
            nearest_r, nearest_c = indices[:, r, c]
            filled_map[r, c] = depth_map[nearest_r, nearest_c]

        # 步骤6：填充其他符合条件的NaN（非闭运算区域）
        remaining_nan = (mask == 0) & ~new_fill_regions
        rows_rest, cols_rest = np.where(remaining_nan & (distances <= max_distance))
        for r, c in zip(rows_rest, cols_rest):
            nearest_r, nearest_c = indices[:, r, c]
            filled_map[r, c] = depth_map[nearest_r, nearest_c]

        # 统计信息
        total_filled = len(rows_new) + len(rows_rest)
        print(f"填充总数: {total_filled}（闭运算建议: {len(rows_new)}，其他: {len(rows_rest)}）")

        return filled_map

    def create_depth_map(self, points_img, depths, if_fill=True, morph_preprocess=True, eps=0.6, min_samples=3,
                         edge_width=0,zoom=1):
        """
        创建深度图，使用DBSCAN处理每个像素的多个点
        :param points_img: 投影后的2D坐标
        :param depths: 对应的深度值
        :param eps: DBSCAN的邻域半径(2cm)
        :param min_samples: DBSCAN的最小样本数
        :return: 深度图
        """
        print("开始创建深度图...")
        print(f"当前图像分辨率为：{self.width}x{self.height}")
        start_time = time.time()

        # 将浮点坐标转换为整数像素坐标
        pixel_coords = np.round(points_img).astype(int)

        # 初始化深度图
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)
        depth_map.fill(np.nan)  # 初始为NaN表示无数据

        # 创建哈希表存储每个像素对应的所有深度值
        pixel_dict = {}
        for (x, y), depth in zip(pixel_coords, depths):
            if 0 <= x < self.width and 0 <= y < self.height:
                if (x, y) not in pixel_dict:
                    pixel_dict[(x, y)] = []
                pixel_dict[(x, y)].append(depth/zoom)

        # 处理每个像素（添加tqdm进度条）
        processed_pixels = 0
        filtered_pixels = 0

        # 使用tqdm显示进度条
        for (x, y), depth_list in tqdm(pixel_dict.items(), desc="处理像素", unit="pixel"):
            processed_pixels += 1
            depth_array = np.array(depth_list)

            # 情况1: 如果深度变化小于阈值，取平均深度
            if np.max(depth_array) - np.min(depth_array) < eps:
                depth_map[y, x] = np.mean(depth_array)

            # 情况2: 使用DBSCAN聚类
            else:
                X = np.column_stack([
                    depth_array,
                    np.random.normal(0, 1, size=len(depth_array))
                ])

                db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = db.labels_

                # 忽略噪声点(label=-1)
                valid_labels = labels[labels != -1]
                if len(valid_labels) == 0:
                    continue

                else:  # 取最大的簇(点数最多的簇)
                    unique_labels, counts = np.unique(valid_labels, return_counts=True)
                    main_cluster_label = unique_labels[np.argmax(counts)]

                    # 取该簇中最近的点的深度
                    cluster_depths = depth_array[labels == main_cluster_label]
                    depth_map[y, x] = np.min(cluster_depths)
                    filtered_pixels += 1

        print(f"\n处理完成! 总像素数: {processed_pixels}, 过滤像素数: {filtered_pixels}")
        print(f"耗时: {time.time() - start_time:.2f}秒")

        if if_fill:
            depth_map = self.fill_nan_with_nearest_neighbor(
                depth_map,
                max_distance=1,
                use_morphology=morph_preprocess,
                edge_width=edge_width
            )

        return depth_map

    def visualize_depth_map(self,depth_map, if_pure=False ,save_path=None):
        """
        可视化深度图
        """
        if if_pure:
            plt.figure(figsize=(12, 6))
            plt.imshow(depth_map, cmap='viridis')
            plt.axis('off')
        else:
          plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
          plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
          plt.figure(figsize=(12, 6))
          plt.imshow(depth_map, cmap='viridis')
          plt.colorbar(label='深度值(m)')
          plt.title(f'精制栅格深度图（{self.current_camera_name}）')
          plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"深度图已保存到 {save_path}")
        plt.close()  # 关闭图形以避免内存泄漏

    def process_single_camera(self, camera_params_path, output_dir, eps, min_samples,R_kind, if_fill, morph_preprocess,
                              edge_width,zoom,if_pure):
        """处理单个相机参数文件"""
        try:
            self.load_camera_parameters(camera_params_path)

            # 1. 将点云投影到图像平面
            points_img, depths, _ = self.project_points_to_image(R_kind)

            # 2. 创建深度图
            depth_map = self.create_depth_map(
                points_img, depths,
                if_fill=if_fill,
                eps=eps,
                min_samples=min_samples,
                morph_preprocess=morph_preprocess,
                edge_width=edge_width,
                zoom=zoom
            )

            # 3. 保存结果
            output_filename = f"{self.current_camera_name}_depth.png"
            output_path = os.path.join(output_dir, output_filename)
            self.visualize_depth_map(depth_map, if_pure=if_pure,save_path=output_path)

            return depth_map
        except Exception as e:
            print(f"处理相机 {os.path.basename(camera_params_path)} 时出错: {e}")
            return None

    def batch_process(self, output_dir,if_pure, eps=0.6, min_samples=3,R_kind="B", if_fill=True, morph_preprocess=False, edge_width=0,zoom=1):
        """批量处理目录中的所有相机参数文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取目录中所有的.json文件
        camera_files = [f for f in os.listdir(self.camera_params_dir) if f.endswith('.json')]
        total_cameras = len(camera_files)
        print(f"\n开始批量处理，共发现 {total_cameras} 个相机参数文件")

        for i, camera_file in enumerate(camera_files, 1):
            camera_path = os.path.join(self.camera_params_dir, camera_file)
            print(f"\n处理进度: {i}/{total_cameras} - {camera_file}")

            self.process_single_camera(
                camera_path, output_dir,
                eps=eps,
                min_samples=min_samples,
                R_kind=R_kind,
                if_fill=if_fill,
                morph_preprocess=morph_preprocess,
                edge_width=edge_width,
                zoom=zoom,
                if_pure=if_pure
            )

        print("\n批量处理完成!")

    def rasterize(self, eps=0.6, min_samples=3, R_kind="B",edge_width=0, if_fill=True,
                  output_path=None, morph_preprocess=False, output_dir=None,zoom=1,if_pure=False):
        """
        执行栅格化流程（支持单文件和批量模式）
        """
        if self.camera_params_dir and output_dir:
            # 批量处理模式
            self.batch_process(
                output_dir=output_dir,
                eps=eps,
                min_samples=min_samples,
                if_fill=if_fill,
                morph_preprocess=morph_preprocess,
                edge_width=edge_width,
                zoom=zoom,
                if_pure=if_pure
            )
        elif self.single_camera_params_path:
            # 单文件处理模式
            self.load_camera_parameters(self.single_camera_params_path)

            # 1. 将点云投影到图像平面
            points_img, depths, _ = self.project_points_to_image(R_kind)

            # 2. 创建深度图
            depth_map = self.create_depth_map(
                points_img, depths,
                if_fill=if_fill,
                eps=eps,
                min_samples=min_samples,
                morph_preprocess=morph_preprocess,
                edge_width=edge_width,
                zoom=zoom
            )

            # 3. 可视化结果并保存
            self.visualize_depth_map(depth_map, if_pure=if_pure,save_path=output_path)

            return depth_map
        else:
            raise ValueError("必须指定相机参数目录（批处理模式）或单个相机参数文件路径")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='点云栅格化工具 - 将3D点云投影到2D图像平面生成深度图',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--point_cloud', '-p', type=str, required=True,
                       help='点云文件路径 (.ply 或 .pts 格式)')
    
    # 模式选择参数（互斥组）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--camera_dir', '-d', type=str,
                           help='相机参数目录路径（批处理模式），包含多个.json文件')
    mode_group.add_argument('--camera_file', '-f', type=str,
                           help='单个相机参数文件路径（单文件模式）')
    
    # 输出参数
    parser.add_argument('--output_dir', '-o', type=str,
                       help='输出目录路径（批处理模式时必需）')
    parser.add_argument('--output_path', type=str,
                       help='输出文件路径（单文件模式时必需）')
    
    # 算法参数（有默认值）
    parser.add_argument('--eps', type=float, default=0.6,
                       help='DBSCAN聚类半径参数（单位：米）')
    parser.add_argument('--min_samples', type=int, default=3,
                       help='DBSCAN最小样本数')
    parser.add_argument('--R_kind', type=str, default='B', choices=['A', 'B'],
                       help='相机外参旋转矩阵R的种类: A(行向量派)或B(列向量派)')
    
    # 填充相关参数
    parser.add_argument('--no_fill', action='store_false', dest='if_fill',
                       help='禁用插值填充（默认启用填充）')
    parser.set_defaults(if_fill=True)
    
    parser.add_argument('--morph_preprocess', action='store_true',
                       help='启用形态学闭运算预处理（默认禁用）')
    parser.add_argument('--edge_width', type=int, default=0,
                       help='边缘保护宽度（像素），0表示禁用')
    
    # 其他参数
    parser.add_argument('--zoom', type=float, default=1.0,
                       help='深度图尺度放缩比例（默认单位是米）')
    parser.add_argument('--pure', action='store_true',
                       help='生成纯深度图（无标题、颜色条、坐标轴）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 参数验证
    if args.camera_dir and not args.output_dir:
        parser.error("批处理模式需要指定 --output_dir 参数")
    
    if args.camera_file and not args.output_path:
        parser.error("单文件模式需要指定 --output_path 参数")
    
    # 检查文件/目录是否存在
    import os
    if not os.path.exists(args.point_cloud):
        parser.error(f"点云文件不存在: {args.point_cloud}")
    
    if args.camera_dir and not os.path.exists(args.camera_dir):
        parser.error(f"相机参数目录不存在: {args.camera_dir}")
    
    if args.camera_file and not os.path.exists(args.camera_file):
        parser.error(f"相机参数文件不存在: {args.camera_file}")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印参数信息
    print("=" * 60)
    print("点云栅格化参数配置")
    print("=" * 60)
    print(f"点云文件: {args.point_cloud}")
    print(f"运行模式: {'批处理' if args.camera_dir else '单文件'}")
    if args.camera_dir:
        print(f"相机参数目录: {args.camera_dir}")
        print(f"输出目录: {args.output_dir}")
    else:
        print(f"相机参数文件: {args.camera_file}")
        print(f"输出文件: {args.output_path}")
    print(f"DBSCAN参数: eps={args.eps}, min_samples={args.min_samples}")
    print(f"旋转矩阵类型: {args.R_kind}")
    print(f"插值填充: {'启用' if args.if_fill else '禁用'}")
    if args.if_fill:
        print(f"形态学预处理: {'启用' if args.morph_preprocess else '禁用'}")
        print(f"边缘保护宽度: {args.edge_width}像素")
    print(f"深度缩放: {args.zoom}")
    print(f"纯深度图: {'是' if args.pure else '否'}")
    print("=" * 60)
    
    # 创建并运行栅格化器
    rasterizer = PointCloudRasterizer(
        point_cloud_path=args.point_cloud,
        camera_params_dir=args.camera_dir if args.camera_dir else None,
        single_camera_params_path=args.camera_file if args.camera_file else None
    )
    
    # 执行处理
    rasterizer.rasterize(
        eps=args.eps,
        min_samples=args.min_samples,
        R_kind=args.R_kind,
        if_fill=args.if_fill,
        morph_preprocess=args.morph_preprocess,
        edge_width=args.edge_width,
        output_dir=args.output_dir,
        output_path=args.output_path,
        zoom=args.zoom,
        if_pure=args.pure
    )


if __name__ == "__main__":
    main() 
