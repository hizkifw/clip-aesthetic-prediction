#!/usr/bin/env python3


def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def main(
    source_folder: str,
    output_file: str = "output.tsv",
    batch_size: int = 32,
):
    import torch
    import clip
    import os
    from tqdm import tqdm
    from modules.mlp import MLP
    from modules.dataset import ImageDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU")

    # CLIP embedding dim is 768 for CLIP ViT L 14
    mlp = MLP(768)

    print("Loading MLP...")
    mlp.load_state_dict(
        torch.load(
            "./modules/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
        )
    )
    mlp.to(device)
    mlp.eval()

    print("Loading CLIP...")
    model_clip, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    # Check existing predictions TSV
    to_skip = []
    tsv_exists = os.path.exists(output_file)
    if tsv_exists:
        with open(output_file, "r") as f:
            next(f)  # skip header
            for line in f:
                to_skip.append(line.split("\t")[0].strip())

        print(f"Skipping {len(to_skip)} images already processed")

    # Prepare dataset
    print("Preparing dataset...")
    dataset = ImageDataset(
        source_folder, filter=lambda x: x not in to_skip, transform=preprocess
    )
    print(f"Found {len(dataset)} images")

    # Prepare dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with open(output_file, "a") as f:
        if not tsv_exists:
            f.write("filename\tavg_rating\n")

        with torch.no_grad():
            for batch in tqdm(loader):
                fidx, img_ok, images = batch["index"], batch["ok"], batch["image"]
                fnames = [dataset.images[x] for x in fidx]

                # Get image embeddings
                image_features = model_clip.encode_image(images.to(device))

                # Normalize image embeddings
                im_emb = normalized(image_features.cpu().detach().numpy())

                # Get predictions
                preds = mlp(torch.from_numpy(im_emb).to(device).float())

                # Save predictions
                for i, pred in enumerate(preds):
                    line = f"{fnames[i]}\t{pred.item() if img_ok[i] == 1 else -1}\n"
                    f.write(line)
                    print(line, end="")


if __name__ == "__main__":
    import typer

    typer.run(main)
