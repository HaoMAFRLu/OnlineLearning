import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def delete_lines_and_save_new_file(input_file_path, output_file_path, lines_to_get):
    # 读取文件的所有行
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # 创建新的内容，跳过要删除的行
    new_lines = [line for idx, line in enumerate(lines) if idx in lines_to_get]

    # 将新的内容写入新的文件
    with open(output_file_path, 'w') as new_file:
        new_file.writelines(new_lines)

def get_lines():
    l = [19,20,21,24] + list(range(28, 55)) + [59,64,65,66,67,79,82,83,98,104,105,109] + list(range(111, 138)) + [142]
    return l

def main():
    input_file = 'params_newton.txt'
    output_file = 'params_newton2.txt'
    root = fcs.get_parent_path(lvl=1)

    input_file_path = os.path.join(root, input_file)
    output_file_path = os.path.join(root, output_file)
    lines_to_get = get_lines()  # 要删除的行号列表（注意：行号从 0 开始）

    delete_lines_and_save_new_file(input_file_path, output_file_path, lines_to_get)

if __name__ == '__main__':
    main()