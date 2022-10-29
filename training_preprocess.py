#!/usr/bin/env python3


def main(
    source_folder: str,
    labels_file: str,
    batch_size: int = 32,
    output_x: str = "x.npy",
    output_y: str = "y.npy",
):
    """
    Prepare a dataset for training.

    Labels file is a TSV with two columns: filename and rating. Image files will
    be loaded from source_folder/filename. The ratings should range between 0
    and 10.
    """

    import numpy as np
    import pandas as pd
    import clip
    import torch
    from tqdm import tqdm
    from PIL import Image
    from modules.util import normalized

    print("Loading CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    print("Loading labels...")
    labels = pd.read_csv(labels_file, sep="\t", index_col=0)

    print("Preparing dataset...")
    x = []
    y = []

    with torch.no_grad():
        for fname, row in tqdm(labels.iterrows(), total=len(labels)):
            rating = row[0]
            try:
                image = (
                    preprocess(Image.open(f"{source_folder}/{fname}"))
                    .unsqueeze(0)
                    .to(device)
                )
            except:
                continue

            embed = model.encode_image(image).cpu().detach().numpy()
            x.append(normalized(embed))
            y_ = np.zeros((1, 1))
            y_[0][0] = rating
            y.append(y_)

    print("Saving...")
    vx = np.vstack(x)
    vy = np.vstack(y)
    np.save(output_x, vx)
    np.save(output_y, vy)

    print("Done")


if __name__ == "__main__":
    import typer

    typer.run(main)
