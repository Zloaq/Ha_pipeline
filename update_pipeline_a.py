#!/opt/anaconda3/envs/p11/bin/python3

import subprocess
import shutil
import tempfile
import re
import os
import sys

class save_param():

    def __init__(self, date='{date}', objectname='{objectname}'):
        
        path_program = os.path.abspath(__file__)
        dir_of_program = os.path.dirname(path_program)
        dir1 = os.path.join(dir_of_program, 'main_a.param')
        dir2 = os.path.join(dir_of_program, 'advanced_a.param')
        if not os.access(dir1, os.R_OK):
            print('main_a.param is not found')
            sys.exit()
        if not os.access(dir2, os.R_OK):
            print('advanced_a.param is not found')
            sys.exit()

        with open(dir1) as f1, open(dir2) as f2:
            lines = f1.readlines() + f2.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            varr = line.split()
            if len(varr)==1:
                print(f'param of {varr[0]} is none.')
                continue

            items = ''
            if varr[1].startswith(('\'', '\"', '(')):
                items = f'{varr[1][0]}\\'
                for index, item in enumerate(varr[1:]):
                    if item.endswith(('\'', '\"', ')')):
                        items = items + item[:-1] + f'\\{item[-1]}' + item[-1]
                        varr[1] = items
                        break
                    items = items + item + ' '
            
            varr[1] = varr[1].replace("{date}", date)
            varr[1] = varr[1].replace("{objectname}", objectname)
            #print(f'aaaaa self.{varr[0]} = {varr[1]}')
            exec(f'self.{varr[0]} = {varr[1]}')
               

def save_directory_names(file_path):
    dir_names = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 対象となるディレクトリ名のパターン
            if re.match(r'^(objfile_dir|matrix_dir|rawdata_infra|rawdata_opt|work_dir)', line):
                key, value = line.split(maxsplit=1)
                dir_names[key] = value.strip()  # keyと値を保存
    return dir_names


def save_shebang(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        if first_line.startswith('#!'):
            return first_line[2:]
    return None


def restore_param(usr_param, def_param):
    attributes1 = vars(usr_param)
    attributes2 = vars(def_param)
    differences = {}
    for key in attributes1.keys() & attributes2.keys():
        if attributes1[key] != attributes2[key]:
            differences[key:attributes1[key]]

    path_program = os.path.abspath(__file__)
    dir_of_program = os.path.dirname(path_program)
    dir1 = os.path.join(dir_of_program, 'main_a.param')
    dir2 = os.path.join(dir_of_program, 'advanced_a.param')
    if not os.access(dir1, os.R_OK):
        print('main_a.param is not found')
        sys.exit()
    if not os.access(dir2, os.R_OK):
        print('advanced_a.param is not found')
        sys.exit()

    for varr in [dir1, dir2]:
        with open(varr) as f1:
            lines = f1.readlines()
        for index, line in enumerate(lines):
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            param_name, param_value = parts[0], parts[1]
            if param_name in attributes1:
                new_value = attributes1[param_name]
                lines[index] = '{:<20} {}\n'.format(param_name, new_value)

        with open(varr, 'w') as f:
            f.writelines(lines)


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
    param1 = save_param()
    save_lib(backup_dir, lib_dir)

    comm = os.path.join(dir_of_program, 'ch_shebang.sh')
    git_pull()
    subprocess.run([comm, shebang], stdout=subprocess.DEVNULL)

    param2 = save_param()
    restore_param(param1, param2)

    restore_directory_names(param_file, saved_dir_names)
    restore_lib_files(backup_dir, lib_dir)
    shutil.rmtree(backup_dir)

    print("Update completed")
