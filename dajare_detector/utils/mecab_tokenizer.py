from logging import getLogger
import os
import subprocess
import shutil

import MeCab

logger = getLogger(__name__)


def _exec_cmd(cmd: str, require_return=True) -> str:
    logger.info(f'execute the command: {cmd}')
    out = subprocess.Popen(cmd,
                           shell=True,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    (stdout, stderr) = out.communicate()
    if out.returncode != 0:
        raise RuntimeError(stderr.decode())

    if not require_return:
        return ''

    results = stdout.decode().split()
    if len(results) == 0:
        raise RuntimeError(
            f'command  "{cmd}" throw error with message "{stderr.decode()}"')
    logger.info(f'the result: {results[0]}')
    return results[0]


def _get_resource_dir(workspace_directory) -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir,
                     workspace_directory))


def _install_dic(workspace_directory):
    logger.info("install neologd")
    work_dir = os.path.join(_get_resource_dir(workspace_directory), 'neologd')
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    _exec_cmd(
        f"git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git {work_dir}",
        require_return=False)
    _exec_cmd(
        f"{os.path.join(work_dir, 'bin/install-mecab-ipadic-neologd')} -n -y",
        require_return=False)


def _get_dicdir():
    return os.path.join(_exec_cmd("mecab-config --dicdir"),
                        'mecab-ipadic-neologd')


def _get_dic_path(workspace_directory,
                  force_reinstall=False,
                  install_if_not_exist=True):
    if force_reinstall:
        _install_dic(workspace_directory)

    if os.path.exists(_get_dicdir()):
        return _get_dicdir()

    if install_if_not_exist:
        _install_dic(workspace_directory)
        return _get_dic_path(install_if_not_exist=False,
                             workspace_directory=workspace_directory)
    return _get_dicdir()


def make_mecab_tagger(workspace_directory, words=False):
    try:
        neologd_dic = _get_dic_path(workspace_directory)
        logger.info(f'neologd path: {neologd_dic}')
        return MeCab.Tagger(f'-Ochasen -d {neologd_dic}')
    except FileNotFoundError:
        return
