import os
from pathlib import Path
import subprocess
import shlex
import logging


def get_dir():
    utils_dirname = os.path.dirname(os.path.abspath(__file__))
    lib_dirname = os.path.abspath(os.path.join(utils_dirname, ".."))
    return lib_dirname


def recursive_listdir(dir_name):
    result = []
    for name in os.listdir(dir_name):
        full_name = os.path.join(dir_name, name)
        if (os.path.isdir(full_name)):
            result += recursive_listdir(full_name)
        else:
            result.append(full_name)
    return result


def shell(cmd):
    logging.info("running shell command: \n {}".format(cmd))
    subprocess.check_call(shlex.split(cmd))


def mkdir_p(dir_path):
    shell("mkdir -p {}".format(dir_path))
    return Path(dir_path)


def mkdir_p_local(relative_dir_path):
    """create folder inside of library if does not exists"""
    local_dir = get_dir()
    abspath = os.path.join(local_dir, relative_dir_path)
    mkdir_p(abspath)
    return abspath
