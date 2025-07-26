import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

plt.style.use('seaborn-v0_8-poster')
plt.rcParams["font.family"] = "Times New Roman"

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
        split_info = info[:-1].split(" ")
        if len(split_info) < 3:
            continue
        p_no, msg_time, msg_type = split_info[0], split_info[1], split_info[-1]
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


def parse_data(data):
    """Parses the JACOBI iteration output into easy to use data.
    """
    keys = list(data.keys())
    weak_key = [i for i in keys if "weak" in i][0]
    strong_keys = sorted([i for i in keys if "strong" in i], key=lambda x: int(x.split("_")[-1]))
    strong_large_key, strong_large_size = strong_keys[1], int(strong_keys[1].split("_")[-1])
    strong_small_key, strong_small_size = strong_keys[0], int(strong_keys[0].split("_")[-1])

    all_parsed_data = {
        "weak": {
            "data": data[weak_key]["weak_size"],
        },
        "strong large": {
            "data": data[strong_large_key],
            "size": strong_large_size,
        },
        "strong small": {
            "data": data[strong_small_key],
            "size": strong_small_size,
        }
    }
    for key, val in all_parsed_data.items():
        parsed_data = {}
        for n_proc, data in val["data"].items():
            parsed_data[n_proc] = parse_output(data)
        all_parsed_data[key]["parsed"] = parsed_data
    return all_parsed_data

def make_plots(base_title: str, res: dict[str, dict[str, float]],
               output_dir: Path, eff: bool = False) -> None:
    x = [int(i) for i in reversed(res.keys())]
    items = list(reversed(res.values()))
    serial = np.array([i["serial"] for i in items])
    mpi = np.array([i["mpi"] for i in items])
    comp = np.array([i["comp"] for i in items])
    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_time.png'
    fig, ax = plt.subplots()
    if eff:
        ax.plot(x, (serial+mpi+comp)[0]/(serial+mpi+comp))
        ax.xaxis.set_ticks(x)
        ax.set_ylabel(f"Efficiency")
    else:
        ax.plot(x, serial, label="Serial")
        ax.plot(x, mpi, label="MPI")
        ax.plot(x, comp, label="Matrix Op.")
        ax.xaxis.set_ticks(x)
        ax.set_ylabel(f"Time (seconds)")
    fig.legend()
    ax.set_title(base_title)
    ax.set_xlabel(f"Processes")
    fig.savefig(saveloc)

    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_prop.png'
    all_sum = (serial + mpi + comp)[::-1]
    fig, ax = plt.subplots()
    ax.stackplot(list(reversed(x)), serial[::-1] / all_sum, mpi[::-1] / all_sum, comp[::-1] / all_sum,
                 labels=["Serial", "MPI", "Matrix Op."])
    ax.set_title(f"{base_title} proportion")
    fig.legend()
    fig.savefig(saveloc)


def plot_time_taken(all_res: dict[str, dict[str, dict[str, float]]], saveloc) -> None:
    saveloc = Path(saveloc)
    all_y = defaultdict(dict)
    for alg, res in all_res.items():
        for scale_type, n_proc_res in res.items():
            items = [i for i in list(n_proc_res.values())]
            serial = np.array([i["serial"] for i in items])
            mpi = np.array([i["mpi"] for i in items])
            comp = np.array([i["comp"] for i in items])
            total_time_taken = serial+mpi+comp
            all_y[alg][scale_type] = total_time_taken
    x = [int(i) for i in list(n_proc_res.keys())]
    fig_s, ax_s = plt.subplots()
    fig_w, ax_w = plt.subplots()
    fig_we, ax_we = plt.subplots()
    ax_s.xaxis.set_ticks(x)
    ax_w.xaxis.set_ticks(x)
    ax_we.xaxis.set_ticks(x)
    for alg, s_res in all_y.items():
        ax_w.plot([float(i) for i in list(all_res['GPU']['weak'].keys())], s_res["weak"], label=alg)
        ax_s.plot(x, s_res["strong large"], label=alg)
        ax_we.plot([float(i) for i in list(all_res['GPU']['weak'].keys())], np.array(s_res["weak"]).min()/np.array(s_res["weak"]), label=alg)
    ax_s.set_title("Strong Scaling Time Taken")
    ax_w.set_title("Weak Scaling Time Taken")
    ax_we.set_title("Weak Scaling Efficiency")
    ax_s.set_xlabel("No. Proceses")
    ax_w.set_xlabel("No. Proceses")
    ax_we.set_xlabel("No. Proceses")
    ax_s.set_ylabel("Total Time Taken (seconds)")
    ax_w.set_ylabel("Total Time Taken (seconds)")
    ax_we.set_ylabel("Efficiency (t(1)/t(N))")
    fig_we.legend()
    fig_s.legend()
    fig_w.legend()
    fig_s.savefig(saveloc/"alg_scaling_strong.png")
    fig_w.savefig(saveloc/"alg_scaling_weak.png")
    fig_we.savefig(saveloc/"alg_scaling_weak_efficiency.png")

def plot_data(parsed_data, compute_mode, out_folder):
    for scaling_type, data in parsed_data.items():
        if scaling_type == "weak":
            make_plots(f"{compute_mode} {scaling_type} efficiency", data["parsed"], out_folder, eff=True)
        else:
            make_plots(f"{compute_mode} {scaling_type} speedup", data["parsed"], out_folder, eff=True)
        make_plots(f"{compute_mode} {scaling_type} scaling", data["parsed"], out_folder)


def main():
    output_path = Path("./figs")
    output_path.mkdir(exist_ok=True)
    data_gpu_path = "./jacobi_gpu.json"
    data_naive_path = "./jacobi_naive.json"

    with open(data_gpu_path, "r") as f:
        data_gpu = json.load(f)
    with open(data_naive_path, "r") as f:
        data_cpu = json.load(f)

    data_gpu = parse_data(data_gpu)
    data_cpu = parse_data(data_cpu)

    plot_data(data_gpu, "GPU", output_path)
    plot_data(data_cpu, "CPU", output_path)

    res = {
        "CPU": data_cpu,
        "GPU": data_gpu
    }
    newdic = defaultdict(lambda: defaultdict(dict))
    for key in res.keys():
        for task_key in res[key].keys():
            newdic[key][task_key] = res[key][task_key]["parsed"]
    plot_time_taken(newdic, output_path)

if __name__ == "__main__":
    main()
