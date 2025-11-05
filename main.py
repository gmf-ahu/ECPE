import time
from sympy import false
import torch.cuda
from eval import Evaluator
from scripts import _C
from utils import *

def main(write=True, search_mode=None):
    start = time.time()
    # write = write
    write = True
    # write = False
    terminal = True

    exp_name = "group"
    exp_scripts = "group"
    log_name = "testgroup"

    config_list, log_path, write = build_experiments(
        exp_name, log_name, write, exp_scripts
    )
    results = []
    for f in config_list:
        config = _C.clone()
        config = set_config(config, f)
        if search_mode:
            terminal = False
            for key in search_mode.keys():
                # assert config[key]
                config[key] = search_mode[key]
        result = evaluate(config, write, terminal)
        results.append(result)
    sum_acc = torch.stack(results, 0).sum()
    print(sum_acc)
    print(
        f"Total Datasets Time: {int((time.time() - start) // 60):d} min "
        f"{((time.time() - start) % 60)} s"
    )
    return results


def evaluate(config, write=None, terminal=True):
    T1 = time.time()
    device = set_device()
    set_seed(config.seed)

    evaluator = Evaluator(config, device)
    results = evaluator.full_evaluation(write, terminal)
    # print(f'running time: {time.time() - T1:.1f}')
    return results


def build_experiments(exp_name="unname", log_name="run", write=None, exp_scripts=""):
    config_list = []
    root = "scripts"
    root = os.path.join(root, exp_scripts)
    file_names = get_file_names_in_subfolders(root)
    for file_name in file_names:
        config_list.append(file_name)
    config_list = [x for x in config_list if "init" not in x and "pycharm" not in x]

    os.makedirs(os.path.join("output", exp_name), exist_ok=True)
    log_path = os.path.join(
        "output",
        exp_name,
        log_name + time.strftime(" %m-%d_%H-%M", time.localtime()) + ".txt",
    )
    write = Log(log_path, write)

    return config_list, log_path, write


if __name__ == "__main__":
    main()
