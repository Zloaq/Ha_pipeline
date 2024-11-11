#!/opt/anaconda3/envs/p11/bin/python3

import subprocess
import shutil
import tempfile
import re
import os

def save_directory_names(file_path):
    dir_names = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 対象となるディレクトリ名のパターン
            if re.match(r'^(objfile_dir|rawdata_infra|rawdata_opt|work_dir)', line):
                key, value = line.split(maxsplit=1)
                dir_names[key] = value.strip()  # keyと値を保存
    return dir_names


def save_shebang(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if first_line.startswith('#!'):
            return first_line[2:]
    return None


def save_lib(backup_dir, lib_dir):
    if os.path.exists(backup_dir):
        # バックアップディレクトリからlibディレクトリにコピー
        shutil.copytree(backup_dir, lib_dir, dirs_exist_ok=True)
        print(f"lib directory restored from {backup_dir}")
    else:
        print(f"Backup directory not found: {backup_dir}")


def restore_lib_files(backup_dir, lib_dir):
    if os.path.exists(backup_dir):
        # バックアップディレクトリからlibディレクトリにコピー
        shutil.copytree(backup_dir, lib_dir, dirs_exist_ok=True)
        print(f"lib directory restored from {backup_dir}")
    else:
        print(f"Backup directory not found: {backup_dir}")



def git_pull():
    try:
        print("Pulling the latest updates from the repository...")
        result = subprocess.run(['git', 'reset', '--hard'], check=True, text=True, capture_output=True)
        print(result.stdout)  # git pull の結果を表示
        result = subprocess.run(['git', 'pull'], check=True, text=True, capture_output=True)
        print(result.stdout)  # git pull の結果を表示
    except subprocess.CalledProcessError as e:
        print(f"Error during git pull: {e.stderr}")


def restore_directory_names(file_path, dir_names):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            key = line.split(maxsplit=1)[0] if line.strip() else ""
            if key in dir_names:
                file.write(f"{key:<16}{dir_names[key]}\n")
            else:
                file.write(line)


if __name__ == "__main__":

    path_program = os.path.abspath(__file__)
    dir_of_program = os.path.dirname(path_program)
    param_file = os.path.join(dir_of_program, 'main_a.param')
    lib_dir = os.path.join(dir_of_program, 'lib')
    backup_dir = tempfile.mkdtemp(prefix="lib_backup_")

    saved_dir_names = save_directory_names(param_file)
    shebang = save_shebang(path_program)
    save_lib(backup_dir, lib_dir)

    comm = os.path.join(dir_of_program, 'ch_shebang.sh')
    git_pull()
    subprocess.run([comm, shebang], stdout=subprocess.DEVNULL)

    restore_directory_names(param_file, saved_dir_names)
    restore_lib_files(backup_dir, lib_dir)
    shutil.rmtree(backup_dir)

    print("Update completed")
