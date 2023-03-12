import os
import sys
from collections import OrderedDict
import os.path as osp


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not any(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def get_file_id(filename):
    if type(filename) == list:
        filename = filename[0]
    return '_'.join((osp.basename(filename).replace('.json', '').split('_')[:-1]))


def file_preprocessing(ann_file_paths):
    file_list = []
    for anno_file_path in ann_file_paths:
        _id = get_file_id(anno_file_path)
        prev = [x for x in file_list if get_file_id(x) == _id]
        if len(prev) > 0:
            prev[0].append(anno_file_path)
        else:
            file_list.append([anno_file_path])
    return file_list
