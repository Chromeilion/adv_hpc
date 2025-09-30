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
        prev_time = 0
        for msg_time, msg_type in msg_list:
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


            prev_time = float(msg_time)

    nproc = max(proc_outs.keys())
    return {"serial": serial / nproc, "mpi": mpi / nproc, "comp": comp / nproc}


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
    x = [int(i)//4 for i in reversed(res.keys())]

    items = list(reversed(res.values()))
    serial = np.array([i["serial"] for i in items])
    mpi = np.array([i["mpi"] for i in items])
    comp = np.array([i["comp"] for i in items])
    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_time.png'
    fig, ax = plt.subplots()
    if eff:
        ax.plot(x, (serial+mpi+comp)[-1]/(serial+mpi+comp))
        ax.xaxis.set_ticks(x)
        ax.set_ylabel(f"Efficiency")
    else:
        ax.plot(x, serial)#, label="Serial")
#        ax.plot(x, mpi, label="MPI")
#        ax.plot(x, comp, label="Matrix Op.")
        ax.xaxis.set_ticks(x)
        ax.set_ylabel(f"Time (seconds)")
    fig.legend()
    ax.set_title(base_title)
    ax.set_xlabel(f"Nodes")
    fig.savefig(saveloc)

    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_prop.png'
    all_sum = (serial + mpi + comp)[::-1]
#    fig, ax = plt.subplots()
#    ax.stackplot(list(reversed(x)), serial[::-1] / all_sum, mpi[::-1] / all_sum, comp[::-1] / all_sum,
#                 labels=["Serial", "MPI", "Matrix Op."])
#    ax.set_title(f"{base_title} proportion")
#    fig.legend()
#    fig.savefig(saveloc)

def plot_data(parsed_data, compute_mode, out_folder):
    # parsed_data is now: {proc: {task_key: [ {serial, mpi, comp}, ... ]}}
    # 1) Average runs per (proc, task_key)
    def avg_records(records):
        if not records:
            return {"serial": 0.0, "mpi": 0.0, "comp": 0.0}
        n = len(records)
        s = sum(r.get("serial", 0.0) for r in records) / n
        m = sum(r.get("mpi", 0.0) for r in records) / n
        c = sum(r.get("comp", 0.0) for r in records) / n
        return {"serial": s, "mpi": m, "comp": c}

    # 2) Reorganize into per-task across procs
    task_to_proc: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for proc, task_map in parsed_data.items():
        for task_key, runs in task_map.items():
            task_to_proc[task_key][proc] = avg_records(runs)

    # 3) For each task, sort procs and plot
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    for task_key, proc_map in task_to_proc.items():
        # sort procs numerically to preserve order in dict
        sorted_procs = sorted(proc_map.keys())
        ordered = {p: proc_map[p] for p in sorted_procs}

        is_weak = "weak" in task_key.lower()
        if is_weak:
            continue
            # Efficiency for weak scaling
            make_plots(f"{compute_mode} {task_key} efficiency", ordered, out_folder, eff=True)
        else:
            # Speedup (via efficiency-style curve) for strong scaling
            make_plots(f"{compute_mode} {task_key} speedup", ordered, out_folder, eff=True)
        # Absolute time and proportions
        make_plots(f"{compute_mode} {task_key} scaling", ordered, out_folder)

def plot_time_taken(all_res: dict[str, dict[str, dict[int, dict[str, float]]]], saveloc) -> None:
    saveloc = Path(saveloc)
    saveloc.mkdir(parents=True, exist_ok=True)

    # Build totals per (alg, task_key) over sorted procs
    totals: dict[str, dict[str, tuple[list[int], np.ndarray]]] = defaultdict(dict)
    for alg, tasks in all_res.items():
        for task_key, proc_map in tasks.items():
            if not proc_map:
                continue
            procs_sorted = sorted(proc_map.keys())
            items = [proc_map[p] for p in procs_sorted]
            serial = np.array([i.get("serial", 0.0) for i in items], dtype=float)
            mpi = np.array([i.get("mpi", 0.0) for i in items], dtype=float)
            comp = np.array([i.get("comp", 0.0) for i in items], dtype=float)
            total = serial + mpi + comp
            totals[alg][task_key] = (procs_sorted, total)

    # Separate weak and strong tasks
    weak_tasks = defaultdict(dict)    # alg -> task_key -> (x, y)
    strong_tasks = defaultdict(dict)  # alg -> task_key -> (x, y)
    for alg, task_map in totals.items():
        for task_key, xy in task_map.items():
            if "weak" in task_key.lower():
                weak_tasks[alg][task_key] = xy
            else:
                strong_tasks[alg][task_key] = xy

    # Strong scaling time (prefer the largest size if multiple strong tasks exist)
    fig_s, ax_s = plt.subplots()
    for alg, task_map in strong_tasks.items():
        if not task_map:
            continue
        # try to select "strong large" by max numeric suffix; fallback to arbitrary stable order
        def size_of(k: str) -> int:
            parts = k.split("_")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return -1
        selected_key = max(task_map.keys(), key=size_of)
        x, y = task_map[selected_key]
        x = np.array(x) / 4
        ax_s.plot(x, y, label=f"{alg} ({selected_key})")
    ax_s.set_title("Strong Scaling Time Taken")
    ax_s.set_xlabel("Nodes")
    ax_s.set_ylabel("Total Time Taken (seconds)")
    ax_s.xaxis.set_ticks(x)
    ax_s.legend()
    fig_s.savefig(saveloc / "alg_scaling_strong.png")

    # Weak scaling time and efficiency (t(1)/t(N))
#    fig_w, ax_w = plt.subplots()
#    fig_we, ax_we = plt.subplots()
#    for alg, task_map in weak_tasks.items():
#        if not task_map:
#            continue
#        # If multiple weak tasks exist, plot each
#        for task_key, (x, y) in task_map.items():
#            ax_w.plot(x, y, label=f"{alg} ({task_key})")
#            eff = (y[0] / y) if len(y) > 0 and y[0] > 0 else np.ones_like(y)
#            ax_we.plot(x, eff, label=f"{alg} ({task_key})")

#    ax_w.set_title("Weak Scaling Time Taken")
#    ax_w.set_xlabel("No. Processes")
#    ax_w.set_ylabel("Total Time Taken (seconds)")
#    ax_w.legend()
#    fig_w.savefig(saveloc / "alg_scaling_weak.png")

#    ax_we.set_title("Weak Scaling Efficiency")
#    ax_we.set_xlabel("No. Processes")
#    ax_we.set_ylabel("Efficiency (t(1)/t(N))")
#    ax_we.legend()
#    fig_we.savefig(saveloc / "alg_scaling_weak_efficiency.png")


def main():
    output_path = Path("./figs")
    output_path.mkdir(exist_ok=True)
    data_dir = Path("./results")

    variations = ["gpu_graphs", "gpu", "naive"]
    procs = [4, 8, 16, 32, 64, 128]

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for var in variations:
        for proc in procs:
            with open(data_dir/f"jacobi_{var}{proc}.json", "r") as f:
                f = json.load(f)
                for key, val in f.items():
                    for test in val:
                        test = parse_output(test)
                        data[var][proc][key].append(test)

    for key, val in data.items():
        plot_data(val, key, output_path)

    # Build input for plot_time_taken:
    # all_res: {algorithm: {task_key: {proc: {serial, mpi, comp}}}}
    all_res = defaultdict(dict)
    for alg, proc_map in data.items():
        task_collect = defaultdict(dict)  # task_key -> {proc -> avg dict}
        for proc, task_runs in proc_map.items():
            for task_key, runs in task_runs.items():
                # average runs for this (alg, proc, task_key)
                if runs:
                    n = len(runs)
                    avg = {
                        "serial": sum(r.get("serial", 0.0) for r in runs) / n,
                        "mpi":    sum(r.get("mpi", 0.0) for r in runs) / n,
                        "comp":   sum(r.get("comp", 0.0) for r in runs) / n,
                    }
                else:
                    avg = {"serial": 0.0, "mpi": 0.0, "comp": 0.0}
                task_collect[task_key][int(proc)] = avg
        all_res[alg] = task_collect


    plot_time_taken(all_res, output_path)

if __name__ == "__main__":
    main()

