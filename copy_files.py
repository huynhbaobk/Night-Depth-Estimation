import os
import shutil

def extract_first_column(file_path):
    with open(file_path, 'r') as file:
        return [line.split()[0] for line in file]

def copy_files(source_folder, destination_folder, file_names):
    for file_name in file_names:
        source_file_path = os.path.join(source_folder, file_name) + ".png"
        destination_file_path = os.path.join(destination_folder, file_name) + ".png"
        # Check if the file exists before copying
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, destination_file_path)
        else:
            print(f"File not found: {source_file_path}")

if __name__ == "__main__":
    file_list_path = "split_files/robotcar/2014-12-09-13-21-02/train_split.txt"
    source_folder = "/media/aiteam/DataAI/depth_datasets/oxford_raw/2014-12-09-13-21-02/rgb"
    destination_folder = "/media/aiteam/DataAI/depth_datasets/oxford_raw/2014-12-09-13-21-02/train_rnw"

    # Extracting file names from the text file
    file_names = extract_first_column(file_list_path)

    print(len(file_names))
    print(file_names[:5])

    # Copying files from the first folder to the second folder
    copy_files(source_folder, destination_folder, file_names)
