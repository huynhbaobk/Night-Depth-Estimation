import os

def list_png_files(root_folder):
    png_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_folder)
                png_files.append(relative_path.replace(os.sep, '/'))
    return png_files

def save_to_txt(png_files, output_file):
    with open(output_file, 'w') as f:
        for i, file_name in enumerate(png_files):
            f.write(file_name + '\n')
            if i < len(png_files) - 1:  # Add "-----" after each folder
                current_folder = file_name.split('/')[0]
                next_folder = png_files[i + 1].split('/')[0]
                if current_folder != next_folder:
                    f.write("-----\n")

if __name__ == "__main__":
    root_folder = "/media/aiteam/DataAI/depth_datasets/epe/epe-depth/depth"
    output_file = "train_epe_depth_split.txt"
    file_list = list_png_files(root_folder)
    save_to_txt(file_list, output_file)
