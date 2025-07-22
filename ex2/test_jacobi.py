import argparse as ap
import subprocess
from os import PathLike, environ
from dataclasses import dataclass
import json
from collections import defaultdict


# Leonardo spec
BOOST_N_CPU_CORES = 32
BOOST_N_GPUS = 4


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

    def run(self, size: int, n_nodes: int, iters, leniance) -> str:
        environ["ACC_NUM_CORES"] = str(self.run_params.n_cores_per_process)
        environ["OPENBLAS_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        environ["GOTO_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        environ["OMP_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        mand_r = subprocess.Popen(
            self.get_command(size, n_nodes, iters, leniance),
            stdout=subprocess.PIPE,
            env=environ,
            text=True
        )
        out = mand_r.communicate()
        return out

    def get_command(self, size: int, n_nodes: int, iters: int, leniance: float) -> list[str]:
        command = [
            "mpirun",
            "-np", str(n_nodes*self.run_params.p_per_node),
            "--map-by",
            f"ppr:{self.run_params.n_processes_per_node}:node:pe={self.run_params.n_cores_per_process}",
            self.bin, str(size), str(iters), str(leniance), "1"
        ]
        return command


def test_weak(run_params: RunParams, runner: MatRunner) -> dict[int, str]:
    # Weak scaling consts
    WEAK_SCALE_RATIO_SIZE = 512
    WEAK_SCALE_RATIO_LENIENCE = 0
    WEAK_SCALE_RATIO_BASE_ITER = 1024

    print("Testing weak scaling", flush=True)
    test_res = defaultdict(dict)
    x = run_params.max_nodes
    proc_list = [run_params.max_nodes]
    while x > 2:
        root = x / 2
        proc_list.append(int(root))
        x = root
    for n_nodes in proc_list:
        size = n_nodes*4*WEAK_SCALE_RATIO_SIZE

        print(f"Testing with {n_nodes} nodes and matrix size {size}", flush=True)
        test_res["weak_size"][n_nodes] = runner.run(
            size, n_nodes, WEAK_SCALE_RATIO_BASE_ITER, WEAK_SCALE_RATIO_LENIENCE)[0]

    return test_res


def test_strong(run_params: RunParams, runner: MatRunner, size: int, iters, lenience) -> dict[int, str]:
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
        test_res[n_nodes] = runner.run(size, n_nodes, iters, lenience)[0]
    return test_res


def main(binary_loc: str, output_file: str, g: bool):
    # Strong scaling consts
    STRONG_SCALE_SIZE_BIG = 2**15
    STRONG_SCALE_SIZE_SMALL = 2**14
    STRONG_SCALE_ITER = 1024
    STRONG_SCALE_LENIENCE = 0

    if g:
        STRONG_SCALE_SIZE_BIG = 80_000
        STRONG_SCALE_SIZE_SMALL = 2**15
    print("Running matrix multiplication scaling tests", flush=True)
    run_params = RunParams()
    runner = MatRunner(binary_loc, run_params)
    res = {
        "weak_N2": test_weak(run_params, runner),
        f"strong_{STRONG_SCALE_SIZE_SMALL}": test_strong(run_params, runner, STRONG_SCALE_SIZE_SMALL, STRONG_SCALE_ITER, STRONG_SCALE_LENIENCE),
        f"strong_{STRONG_SCALE_SIZE_BIG}": test_strong(run_params, runner, STRONG_SCALE_SIZE_BIG, STRONG_SCALE_ITER, STRONG_SCALE_LENIENCE)
    }

    with open(output_file, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-b", "--binary", action="store", required=True)
    parser.add_argument("-o", "--output-file", action="store", required=True)
    parser.add_argument("-g", action="store_true")
    args = parser.parse_args()
    main(args.binary, args.output_file, args.g)