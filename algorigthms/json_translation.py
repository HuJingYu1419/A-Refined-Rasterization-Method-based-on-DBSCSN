import os
import json


def convert_camera_json(input_dir, output_dir):
    """
    将自定义格式的相机JSON文件转换为Open3D兼容格式
    修改：强制将内参主点(cx, cy)转换为整数
    :param input_dir: 输入目录（包含原始JSON文件）
    :param output_dir: 输出目录（保存转换后的JSON文件）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有JSON文件
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取原始JSON文件
        with open(input_path, 'r') as f:
            custom_params = json.load(f)

        # 构建外参向量 (按列优先顺序展开4x4矩阵) - 保持不变
        extrinsic_matrix = [
            custom_params["rotation"][0][0], custom_params["rotation"][1][0], custom_params["rotation"][2][0], 0.0,
            custom_params["rotation"][0][1], custom_params["rotation"][1][1], custom_params["rotation"][2][1], 0.0,
            custom_params["rotation"][0][2], custom_params["rotation"][1][2], custom_params["rotation"][2][2], 0.0,
            custom_params["position"][0], custom_params["position"][1], custom_params["position"][2], 1.0
        ]

        # 计算主点并强制转换为整数（仅修改此处）
        cx = int(custom_params.get("cx", custom_params["width"] / 2))
        cy = int(custom_params.get("cy", custom_params["height"] / 2))

        # 构建内参向量 (按列优先顺序展开3x3矩阵) - 其他值保持不变
        intrinsic_matrix = [
            custom_params["fx"], 0.0, 0.0,
            0.0, custom_params["fy"], 0.0,
            cx, cy, 1.0  # 仅修改cx, cy为整数
        ]

        # 转换为Open3D标准格式（其余部分不变）
        open3d_params = {
            "class_name": "PinholeCameraParameters",
            "extrinsic": extrinsic_matrix,
            "intrinsic": {
                "class_name": "PinholeCameraIntrinsic",
                "height": custom_params["height"],
                "intrinsic_matrix": intrinsic_matrix,
                "width": custom_params["width"]
            },
            "version_major": 1,
            "version_minor": 0
        }

        # 保存转换后的JSON文件
        with open(output_path, 'w') as f:
            json.dump(open3d_params, f, indent=4)

        print(f"转换完成: {filename} -> {output_path} (主点已取整: cx={cx}, cy={cy})")


if __name__ == "__main__":
    # 配置输入输出目录
    input_directory = "./data/camera_parameters/"
    output_directory = "./data/translated_camera_parameters/"

    # 执行转换
    convert_camera_json(input_directory, output_directory)