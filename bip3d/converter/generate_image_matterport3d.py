import os
import zipfile
from argparse import ArgumentParser
from functools import partial

import mmengine


def process_scene(path, output_folder, scene_name):
    """Process single 3Rscan scene."""
    files = list(os.listdir(os.path.join(path, scene_name)))
    for file in files:
        if not file.endswith(".zip"):
            continue
        if file != "sens.zip":
            continue
        with zipfile.ZipFile(os.path.join(path, scene_name, file),
                             'r') as zip_ref:
            if file == "sens.zip":
                zip_ref.extractall(os.path.join(output_folder, scene_name, file[:-4]))
            else:
                zip_ref.extractall(output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_folder',
                        required=True,
                        help='folder of the dataset.')
    parser.add_argument('--output_folder',
                        required=True,
                        help='output folder of the dataset.')
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    mmengine.track_parallel_progress(func=partial(process_scene,
                                                  args.dataset_folder, args.output_folder),
                                              tasks=os.listdir(args.dataset_folder),
                                     nproc=args.nproc)
