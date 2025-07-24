import matplotlib.pyplot as plt
import argparse as ap
from os import PathLike
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


SAVEFILES = ["naive.json", "blas.json", "gpu.json"]

plt.style.use('seaborn-v0_8-paper')
plt.rc('text', usetex=True)
plt.rc('text.latex')
plt.rcParams["font.family"] = "Times New Roman"
plt.xticks(fontsize=14, rotation=90)

def make_plots(base_title: str, res: dict[str, dict[str, float]],
               output_dir: Path) -> None:
    x = [int(i) for i in reversed(res.keys())]
    items = list(reversed(res.values()))
    serial = np.array([i["serial"] for i in items])
    mpi = np.array([i["mpi"] for i in items])
    comp = np.array([i["comp"] for i in items])
    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_time.png'
    fig, ax = plt.subplots()
    ax.plot(x, serial, label="Serial")
    ax.plot(x, mpi, label="MPI")
    ax.plot(x, comp, label="Matrix Mult.")
    ax.xaxis.set_ticks(x)
    fig.legend()
    ax.set_title(base_title)
    ax.set_ylabel(f"Time (seconds)")
    ax.set_xlabel(f"Processes")
    fig.savefig(saveloc)

    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_prop.png'
    all_sum = (serial + mpi + comp)[::-1]
    fig, ax = plt.subplots()
    ax.stackplot(list(reversed(x)), serial[::-1] / all_sum, mpi[::-1] / all_sum, comp[::-1] / all_sum,
                 labels=["Serial", "MPI", "Matrix Mult."])
    ax.set_title(f"{base_title} proportion")
    fig.legend()
    fig.savefig(saveloc)


def plot_time_taken(all_res: dict[str, dict[str, dict[str, float]]], saveloc) -> None:
    saveloc = Path(saveloc)
    all_y = defaultdict(dict)
    for alg, res in all_res.items():
        for scale_type, n_proc_res in res.items():
            items = [parse_output(i) for i in list(n_proc_res.values())]
            serial = np.array([i["serial"] for i in items])
            mpi = np.array([i["mpi"] for i in items])
            comp = np.array([i["comp"] for i in items])
            total_time_taken = serial+mpi+comp
            all_y[alg][scale_type] = total_time_taken
    x = [int(i) for i in list(n_proc_res.keys())]
    fig_s, ax_s = plt.subplots()
    fig_w, ax_w = plt.subplots()
    ax_s.xaxis.set_ticks(x)
    ax_w.xaxis.set_ticks(x)
    for alg, s_res in all_y.items():
        ax_w.plot([int(i) for i in list(all_res['gpu']['weak_N2'].keys())], s_res["weak_N2"], label=alg)
        ax_s.plot(x, s_res["strong_8192"], label=alg)

    ax_s.set_title("Strong Scaling Time Taken")
    ax_w.set_title("Weak Scaling Time Taken")
    ax_s.set_xlabel("No. Proceses")
    ax_w.set_xlabel("No. Proceses")
    ax_s.set_ylabel("Total Time Taken (seconds)")
    ax_w.set_ylabel("Total Time Taken (seconds)")
    fig_s.legend()
    fig_w.legend()
    fig_s.savefig(saveloc/"alg_scaling_strong.png")
    fig_w.savefig(saveloc/"alg_scaling_weak.png")



def parse_output(mat_out: str) -> dict[str, float]:
    serial: float = 0
    mpi: float = 0
    comp: float = 0
    lines = mat_out.split("\n")
    proc_outs = defaultdict(list)
    for line in lines:
        if "|" not in line:
            continue
        info, msg = line.split("|")
        p_no, msg_time, msg_type = info[:-1].split(" ")
        p_no = int(p_no) + 1
        proc_outs[p_no].append((msg_time, msg_type))

    for p_no, msg_list in proc_outs.items():
        prev_msg_type = None
        prev_time = 0
        for msg_time, msg_type in msg_list:
            if prev_msg_type == msg_type:
                continue
            else:
                diff = float(msg_time) - prev_time
                match msg_type:
                    case "s":
                        serial += diff
                    case "p":
                        serial += diff
                    case "c":
                        comp += diff
                    case "m":
                        mpi += diff

                prev_msg_type = msg_type
                prev_time = float(msg_time)

    return {"serial": serial / p_no, "mpi": mpi / p_no, "comp": comp / p_no}


def main(output_folder: PathLike | str):
    out_folder = Path(output_folder)
    out_folder.mkdir(exist_ok=True)

    all_res = {}

    for file in SAVEFILES:
        with open(file, "r") as f:
            res = json.load(f)
        alg_type = file.split('.')[0]
        all_res[alg_type] = res
        for scaling_type, scaling_res in res.items():
            parsed = { key: parse_output(val) for key, val in scaling_res.items() }
            make_plots(f"{alg_type} {scaling_type} scaling", parsed, out_folder)

    plot_time_taken(all_res, out_folder)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-o", "--output-folder", action="store", required=True)
    args = parser.parse_args()
    main(args.output_folder)
