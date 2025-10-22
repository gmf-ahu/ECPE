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

    exp_name = "protolp_im"
    exp_scripts = "protolp_im"
    log_name = "testprotolp_im"

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
        f"{int((time.time() - start) % 60):d} s"
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


def grid_search():
    search_params = {'alpha':0.0,'omega':0.0,'update_rate':0.0,"lam": 0,"zeta":0}
    best_params = [0.0, 0.0, 0.0, 0.0,0.0]
    best_sum_acc = 0.0
    shots = 5
    data_num = 2
    os.makedirs(os.path.join("parameter"), exist_ok=True)
    out_path = os.path.join("parameter",time.strftime(" %m-%d_%H_%M", time.localtime())+f" {shots}shots.txt")
    outlog=Log(out_path,True)
    for a in torch.linspace(0.7, 0.7, 1 ):
        for b in torch.linspace(1, 1, 1):
            for c in torch.linspace(0.6, 0.6, 1):
                for d in torch.linspace(1, 20, 20):
                    for e in torch.linspace(0.7, 0.7, 1):
                        print(best_params)
                        search_params = {"alpha": a, "omega": b, "update_rate": c, "lam": d,"zeta":e}
                        print(search_params)
                        results = main(False, search_params)
                        print(results)
                        sum_acc = torch.stack(results, 0).sum()
                        print(f"Best: {best_sum_acc:.3f} Acc: {sum_acc:.3f}")
                        print(f"best: {best_params}")
                        outlog.info(f"alpha: {a:.2f}\omega: {b:.2f}\tupdate_rate: {c:.2f}\tlam: {d:.2f}\tzeta:{e:.2f}\tresults:{sum_acc*100/data_num:.3f}%")
                        if sum_acc > best_sum_acc:
                            best_sum_acc = sum_acc
                            best_params = [a, b, c, d, e]
                        print('best', best_params)
    print('best', best_params)

def gap_search():
    search_params = {"alpha":0,"lam": 0,"zeta": 0,"entro": True,"pro": True}
    best_params = [0.0, 0.0, 0.0, 0.0]
    best_diff_params = [0.0, 0.0, 0.0, 0.0]
    best_sum_acc = 0
    best_sum_diff = 0
    for a in torch.linspace(0.3, 0.3, 1):
        for b in torch.linspace(10, 10, 1):
            for c in torch.linspace(0.1,0.1,1):
                search_params = {"alpha":a,"lam": b,"zeta": c,"entro": True,"pro": False}
                print(search_params)
                results = main(False, search_params)
                sum_acc_no = torch.stack(results, 0).sum()
                print(sum_acc_no)
                search_params = {"alpha":a,"lam": b,"zeta": c,"entro": True,"pro": True}
                print(search_params)
                results = main(False, search_params)
                sum_acc_en = torch.stack(results, 0).sum()
                print(sum_acc_en)
                sum_acc = sum_acc_no
                sum_acc_diif = sum_acc_en - sum_acc_no
                print(f"Best: {best_sum_acc:.3f} Acc: {sum_acc:.3f} Diff:{sum_acc_diif:.3f}")
                if sum_acc > best_sum_acc:
                    best_sum_acc = sum_acc
                    best_params = search_params
                    print(best_params)
                if sum_acc_diif > best_sum_diff:
                    best_sum_diff = sum_acc_diif
                    best_diff_params = search_params
                    print(best_diff_params)
    print(f"best_params: {best_params}        best_diff_params: {best_diff_params}")


if __name__ == "__main__":
    # grid_search()
    # gap_search()
    main()
