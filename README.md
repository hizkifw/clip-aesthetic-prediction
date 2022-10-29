# clip aesthetic scoring

Based on
[christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor).

This is a wrapper that simplifies usage of the CLIP aesthetic predictor. It goes
through a folder of images, and generates a `.tsv` file with two columns: the
file name, and the avg_rating.

Inference can be interrupted and will automatically resume the next time it's
run.

```sh
# Update submodules
git submodule update --init --recursive

# Create a venv
python3 -m venv venv

# Enter the venv
source ./venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run inference on a directory of images
python ./infer.py ./path/to/images/ \
  --batch-size 32 \
  --output-file output.tsv
```
