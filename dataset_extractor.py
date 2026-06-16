import numpy as np
from pathlib import Path
from PIL import Image

data = np.load("wikiart_hw2.npz", allow_pickle=True)

X = data["X"] # (5400, 32, 32, 3)
y = data["y"]

# class_names
class_names = {
    0: "Baroque",
    1: "Impressionism",
    2: "Cubism",
    3: "Abstract_Expressionism"
}

# extraction root directory
output_dir = Path("wikiart_images")
output_dir.mkdir(exist_ok=True)

# filenames counters
counters = {k: 0 for k in class_names}

for img_array, label in zip(X, y):

    class_dir = output_dir / class_names[int(label)]
    class_dir.mkdir(exist_ok=True)

    idx = counters[int(label)]

    filename = class_dir / f"{class_names[int(label)]}_{idx:04d}.png"

    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(filename)

    counters[int(label)] += 1

print("Extraction complete.")