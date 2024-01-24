import os


def find_folders_with_more_than_n_images(folder_path, n):
    # 获取文件夹中所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 遍历每个子文件夹并检查其中的图片数量
    for subfolder in subfolders:
        # 获取子文件夹中所有文件的列表
        files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]

        # 计算图片数量
        num_images = len(files)

        # 如果图片数量大于n，则输出子文件夹的名字
        if num_images > n:
            print(f"文件夹 '{os.path.basename(subfolder)}' 中有 {num_images} 张图片。")


# 替换 'your_folder_path' 为你实际的文件夹路径
folder_path = '/data/gjx/project/dataset/CAMUS_256_video/tvt/train/img'
folder_path_2 = '/data/gjx/project/dataset/CAMUS_256_video/tvt/test/img'
# 替换 30 为你想要的图片数量阈值
threshold = 30
find_folders_with_more_than_n_images(folder_path, threshold)
print("\n\n\n")
find_folders_with_more_than_n_images(folder_path_2, threshold)

import os


def count_subfolders(folder_path):
    # 获取文件夹中所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 输出子文件夹的数量
    num_subfolders = len(subfolders)
    print(f"文件夹 '{os.path.basename(folder_path)}' 中有 {num_subfolders} 个子文件夹。")


# 替换 'your_folder_path' 为你实际的文件夹路径
folder_path_1 = '/data/gjx/project/dataset/CAMUS_256_video/tvt/train/img'

count_subfolders(folder_path_1)
folder_path_2 = '/data/gjx/project/dataset/CAMUS_256_video/tvt/train/label'

count_subfolders(folder_path_2)
folder_path_3 = '/data/gjx/project/dataset/CAMUS_256_video/tvt/test/img'

count_subfolders(folder_path_3)
folder_path_4 = '/data/gjx/project/dataset/CAMUS_256_video/tvt/test/label'

count_subfolders(folder_path_4)