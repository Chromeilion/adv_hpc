import argparse as ap
import subprocess
from os import PathLike, environ
from dataclasses import dataclass
import json
import math


# Leonardo spec
BOOST_N_CPU_CORES = 32
BOOST_N_GPUS = 4
# Weak scaling consts
WEAK_SIZE = 80000
WEAK_SCALE_RATIO = 100
# Weak scaling consts
STRONG_SCALE_SIZE_BIG = 2**14
STRONG_SCALE_SIZE_SMALL = 2**12

@dataclass
class RunParams:
    max_nodes: int = int(environ["SLURM_NNODES"])
    p_per_node: int = BOOST_N_GPUS
    n_gpus_per_node: int = BOOST_N_GPUS
    n_cores_per_node: int = BOOST_N_CPU_CORES
    n_cores_per_process: int = n_cores_per_node // n_gpus_per_node
    n_processes_per_node: int = BOOST_N_GPUS


class MatRunner:
    def __init__(self, bin_loc: PathLike | str, run_params: RunParams):
        self.bin: str = bin_loc
        self.run_params: RunParams = run_params

    def run(self, size: int, n_nodes: int) -> str:
        mand_r = subprocess.Popen(
            self.get_command(size, n_nodes),
            stdout=subprocess.PIPE,
            env=environ,
            text=True
        )
        return mand_r.communicate()

    def get_command(self, size: int, n_nodes: int) -> list[str]:
        command = [
            "mpirun",
            "-np", str(n_nodes*self.run_params.p_per_node),
            "--map-by",
            f"ppr:{self.run_params.n_processes_per_node}:node:pe={self.run_params.n_cores_per_process}",
            self.bin, str(size)
        ]
        return command


def test_weak(run_params: RunParams, runner: MatRunner) -> dict[int, str]:
    print("Testing weak scaling", flush=True)
    test_res = {}
    x = run_params.max_nodes
    proc_list = [run_params.max_nodes]
    while x > 2:
        root = x / 2
        proc_list.append(int(root))
        x = root
    for n_nodes in proc_list:
        size = n_nodes*WEAK_SCALE_RATIO
        print(f"Testing with {n_nodes} nodes and matrix size {size}", flush=True)
        test_res[n_nodes] = runner.run(size, n_nodes)[0]
    return test_res


def test_strong(run_params: RunParams, runner: MatRunner, size: int) -> dict[int, str]:
    print("Testing strong scaling", flush=True)
    test_res = {}
    x = run_params.max_nodes
    proc_list = [run_params.max_nodes]
    while x > 1:
        root = x / 2
        proc_list.append(int(root))
        x = root
    for n_nodes in proc_list:
        print(f"Testing with {n_nodes} nodes and matrix size {size}", flush=True)
        test_res[n_nodes] = runner.run(size, n_nodes)[0]
    return test_res


def main(binary_loc: str, output_file: str):
    print("Running matrix multiplication scaling tests", flush=True)
    run_params = RunParams()
    runner = MatRunner(binary_loc, run_params)
    res = {
        "weak_N2": test_weak(run_params, runner),
        f"strong_{STRONG_SCALE_SIZE_SMALL}": test_strong(run_params, runner, STRONG_SCALE_SIZE_SMALL),
        f"strong_{STRONG_SCALE_SIZE_BIG}": test_strong(run_params, runner, STRONG_SCALE_SIZE_BIG)
    }

    with open(output_file, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-b", "--binary", action="store", required=True)
    parser.add_argument("-o", "--output-file", action="store", required=True)
    args = parser.parse_args()
    main(args.binary, args.output_file)