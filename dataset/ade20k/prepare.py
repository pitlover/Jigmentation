import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    dataset_dir = os.path.join("/data/Datasets/ADE20K", "ADEChallengeData2016")

    for name in ["training", "validation"]:
        annotation_dir = os.path.join(dataset_dir, "annotations", name)
        output_dir = os.path.join(dataset_dir, "annotations_detectron2", name)
        os.makedirs(output_dir, exist_ok=True)
        for file in tqdm.tqdm(list(Path(annotation_dir).iterdir())):
            output_file = os.path.join(output_dir, file.name)
            convert(file, output_file)
