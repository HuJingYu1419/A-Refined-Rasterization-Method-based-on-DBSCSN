import json
import os


def split_camera_parameters(input_file, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取输入文件
    with open(input_file, 'r') as f:
        cameras = json.load(f)

    # 处理每个相机参数
    for camera in cameras:
        # 生成输出文件名
        output_file = os.path.join(output_dir, f"camera_{camera['id']:03d}.json")

        # 写入单个相机参数
        with open(output_file, 'w') as f:
            json.dump(camera, f, indent=4)

        print(f"Saved camera {camera['id']} to {output_file}")


if __name__ == "__main__":
    # 输入文件和输出目录
    input_json = "./data/cameras.json"
    output_directory = "./data/camera_parameters/"

    # 执行分割
    split_camera_parameters(input_json, output_directory)