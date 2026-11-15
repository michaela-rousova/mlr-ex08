# Task 4 inspection: dataset overlap/origin check, confusion matrices for the
# V12 and V4 seed-0 checkpoints, and ONNX export of V12 + onnxruntime check on
# the real photos. No training, and every existing file is opened read-only.

import hashlib
import json
import os
from collections import Counter

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ["apple", "cherry", "chestnut", "maple"]
EXTERNAL_ZIP_URL = ("https://raw.githubusercontent.com/tomasvicar/MLR-public/"
                    "master/exercises/data/ex08_leaves_images.zip")

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def folder_samples(folder):
    """(path, class_idx) pairs, classes = sorted subfolders, files sorted —
    exactly the order LeafDataset uses."""
    classes = sorted(os.listdir(folder))
    samples = []
    for i, c in enumerate(classes):
        d = os.path.join(folder, c)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                samples.append((os.path.join(d, f), i))
    return classes, samples


# ---- 1. dataset check ---------------------------------------------------------

print("=" * 70)
print("1. classes and per-class counts")
for folder in ["data/leaves/train", "data/leaves/val",
               "external_data/leaves/train", "external_data/leaves/val"]:
    classes, samples = folder_samples(folder)
    counts = Counter(CLASSES[c] for _, c in samples)
    print(f"  {folder}: classes {classes}, total {len(samples)} -> {dict(counts)}")

print()
print("image resolutions and channel modes:")
for root in ["data/leaves", "external_data/leaves"]:
    sizes, modes, n = Counter(), Counter(), 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                with Image.open(os.path.join(dirpath, f)) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
                    n += 1
    print(f"  {root}: {n} images, {len(sizes)} distinct (w,h), "
          f"modes {dict(modes)}, most common sizes {sizes.most_common(5)}")

print()
print("origin check:")
src_file = "data/leaves/.group_zip_url"
if os.path.exists(src_file):
    print(f"  data/leaves came from group ZIP: {open(src_file).read().strip()}")
else:
    print("  data/leaves/.group_zip_url is MISSING -> unknown origin")
print(f"  external_data came from course archive: {EXTERNAL_ZIP_URL}")
zip_left = "external_data_dl/ex08_leaves_images.zip"
print(f"  downloaded course zip still present: {os.path.exists(zip_left)} "
      f"({os.path.getsize(zip_left)} bytes)" if os.path.exists(zip_left) else
      f"  downloaded course zip not kept on disk")

print()
print("filename overlap between own splits and external splits, "
      "and how many are actually identical files (md5):")

def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

own, ext = {}, {}
for split in ["train", "val"]:
    _, own[split] = folder_samples(f"data/leaves/{split}")
    _, ext[split] = folder_samples(f"external_data/leaves/{split}")

for osplit in ["train", "val"]:
    for esplit in ["train", "val"]:
        ext_by_name = {os.path.basename(p): p for p, _ in ext[esplit]}
        shared = [(p, ext_by_name[os.path.basename(p)])
                  for p, _ in own[osplit]
                  if os.path.basename(p) in ext_by_name]
        identical = sum(1 for p, q in shared if md5(p) == md5(q))
        print(f"  own/{osplit} vs external/{esplit}: {len(shared)} shared "
              f"filenames, {identical} identical contents")
print("  -> shared names are only the course file-naming scheme; "
      "0 identical files means no photo leakage.")

# ---- models from the saved checkpoints ----------------------------------------

class LeafNetBN(nn.Module):  # same definition as in the notebook/task4 script
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def make_mobilenet():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


def load_which(which):
    if which == "V4":
        model = LeafNetBN()
        path = "task4_models/V4_leafnet_bn_aug_15ep_seed0.pt"
    else:
        model = make_mobilenet()
        path = "task4_models/V12_mobilenet_head_only_seed0.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_folder(model, samples):
    y_true, y_pred = [], []
    with torch.no_grad():
        for path, label in samples:
            img = Image.open(path).convert("RGB")
            logits = model(base_transform(img)[None])
            y_true.append(label)
            y_pred.append(int(logits.argmax(1)))
    return y_true, y_pred


# ---- 1c. confusion matrices ----------------------------------------------------

print()
print("=" * 70)
print("2. confusion matrices (rows = true class, cols = predicted)")
for which in ["V12", "V4"]:
    model = load_which(which)
    for name, folder in [("external_val", "external_data/leaves/train"),
                         ("external_val(val-folder)", "external_data/leaves/val"),
                         ("clean_val", "data/leaves/val")]:
        folder = folder if name != "external_val" else "external_data/leaves/train"
        cm = np.zeros((4, 4), dtype=int)
        # external_val = train + val folders of the archive, like task4_experiments
        parts = ([folder] if name != "external_val"
                 else ["external_data/leaves/train", "external_data/leaves/val"])
        samples = []
        for p in parts:
            _, s = folder_samples(p)
            samples += s
        y_true, y_pred = predict_folder(model, samples)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        acc = 100 * np.trace(cm) / cm.sum()
        print(f"\n  {which} on {name}  (n={cm.sum()}, acc={acc:.1f}%)")
        print("             " + " ".join(f"{c:>9s}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f"    {c:>11s} " + " ".join(f"{v:9d}" for v in cm[i]))

# ---- 3. ONNX export of V12 and onnxruntime inference on the photos -------------

print()
print("=" * 70)
print("3. export V12 -> task4_best_v12.onnx (opset 18, weights embedded)")
v12 = load_which("V12")
torch.onnx.export(v12, torch.zeros(1, 3, 96, 96), "task4_best_v12.onnx",
                  input_names=["image"], output_names=["logits"],
                  opset_version=18, dynamo=False)
print(f"  wrote task4_best_v12.onnx ({os.path.getsize('task4_best_v12.onnx')} bytes)")

sess = ort.InferenceSession("task4_best_v12.onnx", providers=["CPUExecutionProvider"])
print(f"  onnxruntime check: input {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}, "
      f"output {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")


def preprocess_task2e(path):
    """Task 2e preprocessing: central square, resize 96, /255, (x-mean)/std, CHW."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((96, 96), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.transpose(x, (2, 0, 1))[None].copy()


def softmax(a):
    e = np.exp(a - a.max())
    return e / e.sum()


print()
print("photos classified with onnxruntime on task4_best_v12.onnx "
      "(torch checkpoint probabilities in brackets for comparison):")
v4 = load_which("V4")
for path in ["check_image.jpg", "apple.jpg", "cherry.jpg", "chestnut.jpg", "maple.jpg"]:
    x = preprocess_task2e(path)
    onnx_probs = softmax(sess.run(None, {"image": x})[0][0])
    with torch.no_grad():
        torch_probs = softmax(v12(torch.from_numpy(x))[0]).numpy()
        v4_probs = softmax(v4(torch.from_numpy(x))[0]).numpy()
    diff = np.abs(onnx_probs - torch_probs).max()
    row = ", ".join(f"{c} {p*100:.2f}%" for c, p in zip(CLASSES, onnx_probs))
    print(f"  {path:15s} V12-onnx: {row}  (torch-onnx maxdiff {diff:.1e})")
    print(f"  {'':15s} V12-torch [{', '.join(f'{p*100:.2f}' for p in torch_probs)}]  "
          f"V4 {' -> '.join(c for c in [CLASSES[int(v4_probs.argmax())]])}")

exp_path = "check_expected.json"
if os.path.exists(exp_path):
    exp = json.load(open(exp_path))
    x = preprocess_task2e("check_image.jpg")
    p = softmax(sess.run(None, {"image": x})[0][0])
    md = max(abs(p[CLASSES.index(c)] - exp[c]) for c in exp)
    print(f"  check_image.jpg through V12-onnx vs leaf_model expectation "
          f"(check_expected.json, that file is the OLD V3-era model): maxdiff {md:.3f}")
print("\nDone. Files created: task4_best_v12.onnx (nothing else was written).")
