import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Optional


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        transform: Optional[Callable[[Image], Any]] = None,
        filter: Optional[Callable[[str], bool]] = None
    ):
        self.root = root
        self.transform = transform if transform is not None else lambda x: x
        self.filter = filter if filter is not None else lambda x: True
        self.images = [x for x in os.listdir(root) if self.filter(x)]
        self.dummy = self.transform(Image.new("RGB", (64, 64)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, self.images[idx])
            return {
                "index": idx,
                "ok": 1,
                "image": self.transform(Image.open(img_path)),
            }
        except:
            return {
                "index": idx,
                "ok": 0,
                "image": self.dummy,
            }
