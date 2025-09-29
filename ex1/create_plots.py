import matplotlib.pyplot as plt
import argparse as ap
from os import PathLike
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


# This script is updated to read the same data format as make_plots.py:
# - Reads ./results/jacobi_{variation}{proc}.json for variations and process counts
# - Aggregates multiple runs per (variation, proc, task_key)
# - Generates per-task plots and overall comparison plots

plt.style.use('seaborn-v0_8-paper')
plt.rc('text', usetex=True)
plt.rc('text.latex')
plt.rcParams["font.family"] = "Times New Roman"
plt.xticks(fontsize=14, rotation=90)

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

    nproc = max(proc_outs.keys()) if proc_outs else 1
    return {"serial": serial / nproc, "mpi": mpi / nproc, "comp": comp / nproc}

def make_plots(base_title: str, res: dict[str, dict[str, float]],
               output_dir: Path, eff: bool = False) -> None:
    # x-axis is number of processes (nodes) as integers, sorted ascending
    keys_sorted = sorted(int(k) for k in res.keys())
    x = keys_sorted

    items = [res[k] if k in res else res[k] for k in keys_sorted]
    serial = np.array([i.get("serial", 0.0) for i in items])
    mpi = np.array([i.get("mpi", 0.0) for i in items])
    comp = np.array([i.get("comp", 0.0) for i in items])
    x = np.array(x) / 4
    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_time.png'
    fig, ax = plt.subplots()
    if eff:
        denom = serial + mpi + comp
        ref = denom[0] if len(denom) > 0 else 1.0
        eff_y = np.divide(ref, denom, out=np.ones_like(denom), where=denom != 0)
        ax.plot(x, eff_y)
        ax.xaxis.set_ticks(x)
        ax.set_ylabel("Efficiency")
    else:
        ax.plot(x, serial, label="Serial")
        ax.plot(x, mpi, label="MPI")
        ax.plot(x, comp, label="Matrix Op.")
        ax.xaxis.set_ticks(x)
        ax.set_ylabel("Time (seconds)")
    fig.legend()
    ax.set_title(base_title)
    ax.set_xlabel("Nodes")
    fig.savefig(saveloc)

    saveloc = output_dir/f'{(base_title.replace(" ", "_"))}_prop.png'
    total = serial + mpi + comp
    total_safe = np.where(total == 0, 1.0, total)
    fig, ax = plt.subplots()
    ax.stackplot(x, serial / total_safe, mpi / total_safe, comp / total_safe,
                 labels=["Serial", "MPI", "Matrix Op."])
    ax.set_title(f"{base_title} proportion")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Proportion")
    ax.xaxis.set_ticks(x)
    fig.legend()
    fig.savefig(saveloc)

def plot_time_taken(all_res: dict[str, dict[str, dict[int, dict[str, float]]]], saveloc) -> None:
    saveloc = Path(saveloc)
    saveloc.mkdir(parents=True, exist_ok=True)

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

    weak_tasks = defaultdict(dict)
    strong_tasks = defaultdict(dict)
    for alg, task_map in totals.items():
        for task_key, xy in task_map.items():
            if "weak" in task_key.lower():
                weak_tasks[alg][task_key] = xy
            else:
                strong_tasks[alg][task_key] = xy

    fig_s, ax_s = plt.subplots()
    for alg, task_map in strong_tasks.items():
        if not task_map:
            continue
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

def main(output_folder: PathLike | str):
    out_folder = Path(output_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    # Configuration consistent with make_plots.py
    data_dir = Path("./results")
    variations = ["blas", "gpu", "naive"]
    procs = [4, 8, 16, 32, 64, 128]

    # data[variation][proc][task_key] -> list of parsed dicts
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for var in variations:
        for proc in procs:
            path = data_dir / f"{var}{proc}.json"
            if not path.exists():
                continue
            with open(path, "r") as f:
                content = json.load(f)
                for task_key, runs in content.items():
                    for run in runs:
                        parsed = parse_output(run)
                        data[var][proc][task_key].append(parsed)

    # Per-variation plotting
    for variation, proc_map in data.items():
        # Reorganize into per-task across procs with averages
        task_to_proc: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        for proc, task_runs in proc_map.items():
            for task_key, runs in task_runs.items():
                if runs:
                    n = len(runs)
                    avg = {
                        "serial": sum(r.get("serial", 0.0) for r in runs) / n,
                        "mpi":    sum(r.get("mpi", 0.0) for r in runs) / n,
                        "comp":   sum(r.get("comp", 0.0) for r in runs) / n,
                    }
                else:
                    avg = {"serial": 0.0, "mpi": 0.0, "comp": 0.0}
                task_to_proc[task_key][int(proc)] = avg

        # For each task, produce efficiency/speedup and scaling plots
        for task_key, proc_map_avg in task_to_proc.items():
            # Ensure ascending x order
            proc_map_avg_sorted = {p: proc_map_avg[p] for p in sorted(proc_map_avg)}
            is_weak = "weak" in task_key.lower()
            if is_weak:
                ...
#                make_plots(f"{variation} {task_key} efficiency", proc_map_avg_sorted, out_folder, eff=True)
            else:
                make_plots(f"{variation} {task_key} speedup", proc_map_avg_sorted, out_folder, eff=True)
            make_plots(f"{variation} {task_key} scaling", proc_map_avg_sorted, out_folder)

    # Build input for overall time-taken plots
    all_res = defaultdict(dict)  # alg -> task_key -> {proc -> avg dict}
    for alg, proc_map in data.items():
        task_collect = defaultdict(dict)
        for proc, task_runs in proc_map.items():
            for task_key, runs in task_runs.items():
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

    plot_time_taken(all_res, out_folder)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-o", "--output-folder", action="store", required=True)
    args = parser.parse_args()
    main(args.output_folder)
