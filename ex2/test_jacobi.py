import argparse as ap
import subprocess
from os import PathLike, environ
from dataclasses import dataclass
import json


# Leonardo spec
BOOST_N_CPU_CORES = 32
BOOST_N_GPUS = 4


@dataclass
class RunParams:
    n_nodes: int = int(environ["SLURM_NNODES"])
    n_tasks: int = int(environ["SLURM_NTASKS"])
    p_per_node: int = BOOST_N_GPUS
    n_gpus_per_node: int = BOOST_N_GPUS
    n_cores_per_node: int = BOOST_N_CPU_CORES
    n_cores_per_process: int = n_cores_per_node // n_gpus_per_node
    n_processes_per_node: int = BOOST_N_GPUS


class MatRunner:
    def __init__(self, bin_loc: PathLike | str, run_params: RunParams):
        self.bin: str = bin_loc
        self.run_params: RunParams = run_params

    def run(self, size: int, n_tasks: int, iters, leniance) -> str:
        environ["ACC_NUM_CORES"] = str(self.run_params.n_cores_per_process)
        environ["OPENBLAS_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        environ["GOTO_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        environ["OMP_NUM_THREADS"] = str(self.run_params.n_cores_per_process)
        mand_r = subprocess.Popen(
            self.get_command(size, n_tasks, iters, leniance),
            stdout=subprocess.PIPE,
            env=environ,
            text=True
        )
        out = mand_r.communicate()
        return out

    def get_command(self, size: int, n_tasks: int, iters: int, leniance: float) -> list[str]:
        command = [
            "mpirun",
            "-np", str(n_tasks),
            "--map-by",
            f"ppr:{self.run_params.n_processes_per_node}:node:pe={self.run_params.n_cores_per_process}",
            self.bin, str(size), str(iters), str(leniance), "1"
        ]
        return command


def test_weak(run_params: RunParams, runner: MatRunner) -> list[str]:
    # Weak scaling consts
    WEAK_SCALE_RATIO_SIZE = 512
    WEAK_SCALE_RATIO_LENIENCE = 0
    WEAK_SCALE_RATIO_BASE_ITER = 1024

    print("Testing weak scaling", flush=True)
    test_res = []
    size = run_params.n_nodes*4*WEAK_SCALE_RATIO_SIZE
    print(f"Testing with {run_params.n_tasks} tasks and matrix size {size}", flush=True)
    test_res.append(runner.run(
        size, run_params.n_tasks, WEAK_SCALE_RATIO_BASE_ITER, WEAK_SCALE_RATIO_LENIENCE)[0])

    return test_res


def test_strong(run_params: RunParams, runner: MatRunner, size: int, iters, lenience) -> list[str]:
    print("Testing strong scaling", flush=True)
    test_res = []
    print(f"Testing with {run_params.n_nodes} nodes and matrix size {size}", flush=True)
    no_tests = 5
    for test_no in range(no_tests):
        print(f"Runing test {test_no} of {no_tests}", flush=True)
        test_res.append(runner.run(size, run_params.n_tasks, iters, lenience)[0])
    return test_res


def main(binary_loc: str, output_file: str, g: bool):
    # Strong scaling consts
    STRONG_SCALE_SIZE_BIG = 2**15
    STRONG_SCALE_SIZE_SMALL = 2**14
    STRONG_SCALE_ITER = 2048
    STRONG_SCALE_LENIENCE = 0

    if g:
        STRONG_SCALE_SIZE_BIG = 80_000
        STRONG_SCALE_SIZE_SMALL = 2**15
    print("Running matrix multiplication scaling tests", flush=True)
    run_params = RunParams()
    runner = MatRunner(binary_loc, run_params)
    res = {
#        "weak_N2": test_weak(run_params, runner),
        f"strong_{STRONG_SCALE_SIZE_SMALL}": test_strong(run_params, runner, STRONG_SCALE_SIZE_SMALL, STRONG_SCALE_ITER, STRONG_SCALE_LENIENCE),
        f"strong_{STRONG_SCALE_SIZE_BIG}": test_strong(run_params, runner, STRONG_SCALE_SIZE_BIG, STRONG_SCALE_ITER, STRONG_SCALE_LENIENCE)
    }
    output_file += str(run_params.n_tasks)+".json"
    with open(output_file, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-b", "--binary", action="store", required=True)
    parser.add_argument("-o", "--output-file", action="store", required=True)
    parser.add_argument("-g", action="store_true")
    args = parser.parse_args()
    main(args.binary, args.output_file, args.g)