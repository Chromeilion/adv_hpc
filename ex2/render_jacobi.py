from os import environ, PathLike, cpu_count
import subprocess
import argparse as ap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm
from multiprocessing import Pool
from functools import partial
import tempfile


SAVELOC = "./frames"
VIDFILE = "video.mp4"
N_FRAMES = 300
IM_SIZE = 512
LENIENCE = 0.0001
MAX_ITERS = 100000


class JacRunner:
    def __init__(self, bin_loc: PathLike | str):
        self.bin: str = bin_loc

    def run(self, size: int, iters, leniance, cwd) -> str:
        mand_r = subprocess.Popen(
            [self.bin, str(size), str(iters), str(leniance), "1"],
            stdout=subprocess.PIPE,
            env=environ,
            text=True,
            cwd=cwd
        )
        out = mand_r.communicate()
        return out


def create_video(savefolder, vid_out):
    command = f"ffmpeg -y -framerate 30 -pattern_type glob -i '{savefolder}/*.png' -c:v libx264 -pix_fmt yuv420p {vid_out}"
    res = subprocess.Popen(command, shell=True).communicate()
    return res


def create_frame(data):
    data = data / data.max()
    cm = plt.get_cmap('inferno')
    colored_data = cm(data)
    return Image.fromarray((colored_data[:, :, :3] * 255).astype(np.uint8))


def save_frame(saveloc, bin_loc, n_iters):
    runner = JacRunner(bin_loc)
    n_iters = (MAX_ITERS / N_FRAMES) * n_iters
    with tempfile.TemporaryDirectory() as tmp:
        runner.run(IM_SIZE, n_iters, LENIENCE, tmp)
        with open(Path(tmp)/"jacobi_output.txt", "r") as f:
            lines = f.readlines()
        data = np.array(
            [[float(j.strip()) for j in i.split(" ") if j.strip()] for i in lines]
        )
        return data, n_iters


def main(bin_loc: str):
    saveloc = Path(SAVELOC)
    saveloc.mkdir(exist_ok=True)
    save_frame_p = partial(save_frame, saveloc, bin_loc)
    iters = list(range(N_FRAMES))

    with Pool(1) as p:
        for data, n_iters in tqdm.tqdm(p.imap(save_frame_p, iters), total=len(iters)):
            frame_path = saveloc/(str(n_iters).rjust(5, "0")+".png")
            im = create_frame(data)
            im.save(frame_path)


    create_video(saveloc, VIDFILE)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-b", "--binary", action="store", required=True)
    args = parser.parse_args()
    main(args.binary)
