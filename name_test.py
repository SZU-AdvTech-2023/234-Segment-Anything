import os

def get_subfolders(folder):
    return [f.path for f in os.scandir(folder) if f.is_dir()]

def check_corresponding_folders(folder1, folder2):
    # 获取两个文件夹中的小文件夹列表
    subfolders1 = get_subfolders(folder1)
    subfolders2 = get_subfolders(folder2)

    # 检查两个列表的长度是否一致
    if len(subfolders1) != len(subfolders2):
        print("两个文件夹中的小文件夹数量不一致")
        return

    # 对两个列表进行排序
    subfolders1.sort()
    subfolders2.sort()

    # 遍历排序后的列表，检查对应位置上的小文件夹是否具有相同的命名
    for folder1, folder2 in zip(subfolders1, subfolders2):
        name1 = os.path.basename(folder1)
        name2 = os.path.basename(folder2)

        if name1 != name2:
            print(f"文件夹命名不一致: {name1} vs {name2}")
        else:
            print(f"文件夹命名一致: {name1}")

# 替换 'folder1_path' 和 'folder2_path' 为你实际的文件夹路径
# folder1_path = '/data/gjx/project/dataset/CAMUS_256_video/tvt/train/img'
# folder2_path = '/data/gjx/project/dataset/CAMUS_256_video/tvt/train/label'
folder1_path = '/data/gjx/project/dataset/CAMUS_256_video/tvt/test/img'
folder2_path = '/data/gjx/project/dataset/CAMUS_256_video/tvt/test/label'
check_corresponding_folders(folder1_path, folder2_path)
