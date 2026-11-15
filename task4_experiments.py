# Task 4: improving the leaf classifier. One-change-at-a-time ablation across
# 3 seeds, with a clean / camera-like / external validation setup.
# Everything below is taken from ex08_student.ipynb (Task 3 pipeline):
# LeafDataset, the transforms, LeafNet, LeafNetBN, the training loop style.

import json
import os
import statistics
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEEDS = [0, 1, 2]

# Crash-resume: after every finished (variant, seed) the results are written to
# task4_progress.json. A restarted run skips seeds already in there. If the run
# ever finishes completely, the file is deleted at the end (delete it by hand to
# force a full re-run).
PROGRESS_FILE = "task4_progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_progress():
    with open(PROGRESS_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(PROGRESS, f, default=str)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

PROGRESS = load_progress()
GROUP_ZIP_URL = "https://drive.google.com/uc?id=10o-TJxunheUz-C0qQaRkYYVNGhNKy-A_"
EXTERNAL_ZIP_URL = "https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/ex08_leaves_images.zip"
TIME_BUDGET_S = 3600  # ~1 hour warning threshold from the task statement


# ---- the dataset class from the notebook, unchanged -------------------------

class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        self.classes = sorted(os.listdir(folder))
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(folder, class_name)
            if os.path.isdir(class_dir):
                files = sorted(os.listdir(class_dir))
                for f in files:
                    self.samples.append((os.path.join(class_dir, f), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, class_idx = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_idx


# ---- the two networks from the notebook, unchanged --------------------------

class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class LeafNetBN(nn.Module):
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
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


# V9: the single change is Dropout(0.3) before the last linear layer.
class LeafNetBNDrop(LeafNetBN):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4)
        )


# ---- transforms --------------------------------------------------------------

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

aug_transform = transforms.Compose([  # the notebook's augmentation, scale=(0.6, 1.0)
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

aug_strong_transform = transforms.Compose([  # V10: only scale is changed
    transforms.RandomResizedCrop(96, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

aug_color_transform = transforms.Compose([  # V10b: only ColorJitter is added
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---- data download / preparation (guarded: nothing is re-downloaded) --------

def prepare_group_data():
    if os.path.isdir("data/leaves"):
        return
    import gdown  # same download the notebook uses
    gdown.download(url=GROUP_ZIP_URL, output="data_group.zip", quiet=True)
    with zipfile.ZipFile("data_group.zip") as archive:
        archive.extractall("data")
    os.remove("data_group.zip")


def prepare_external_data():
    if os.path.isdir("external_data/leaves"):
        return
    os.makedirs("external_data_dl", exist_ok=True)
    zip_path = "external_data_dl/ex08_leaves_images.zip"
    urllib.request.urlretrieve(EXTERNAL_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall("external_data")


prepare_group_data()
prepare_external_data()

train_ds_plain = LeafDataset("data/leaves/train", base_transform)
clean_val = LeafDataset("data/leaves/val", base_transform)
print(f"group data: train {len(train_ds_plain)}, clean_val {len(clean_val)}")
assert len(clean_val) == 62, "clean_val should have 62 images (1 image = 1.61%)"

# external_val: last year's photos (train+val folders of the archive, here ALL
# 820 images are used strictly for evaluation in V1-V13, never for training).
external_val = ConcatDataset([
    LeafDataset("external_data/leaves/train", base_transform),
    LeafDataset("external_data/leaves/val", base_transform),
])
print(f"external_val: {len(external_val)} images")

# camera-like validation: 5 perturbed copies of each of the 62 val images,
# generated once with a fixed seed (999) so every variant sees the same set.
cam_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
cam_images, cam_labels = [], []
torch.manual_seed(999)
for path, label in clean_val.samples:
    img = Image.open(path).convert("RGB")
    for _ in range(5):
        cam_images.append(cam_transform(img))
        cam_labels.append(label)
cam_val = TensorDataset(torch.stack(cam_images), torch.tensor(cam_labels))
print(f"cam_val: {len(cam_val)} images (62 x 5, generated with seed 999)")

os.makedirs("task4_models", exist_ok=True)


# ---- training ----------------------------------------------------------------

def make_mobilenet():
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


def accuracy(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct, len(dataset)


def train_variant(name, make_model, train_ds, epochs, lr, weight_decay=0.0,
                  cosine=False, head_only=False, finetune_all=False):
    results = []
    for seed in SEEDS:
        done = PROGRESS.get(name, {}).get(str(seed))
        if done:
            results.append(done)
            print(f"  {name} seed {seed}: resumed from {PROGRESS_FILE}")
            continue
        torch.manual_seed(seed)
        model = make_model().to(device)
        if head_only:
            for p in model.features.parameters():
                p.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
        elif finetune_all:
            trainable = list(model.parameters())
        else:
            trainable = list(model.parameters())
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
        loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        criterion = nn.CrossEntropyLoss()
        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            if head_only:
                model.features.eval()  # keep BatchNorm stats frozen
            loss_sum = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            if scheduler:
                scheduler.step()
        epoch_time = (time.time() - t0) / epochs
        clean_c, clean_n = accuracy(model, clean_val)
        cam_c, cam_n = accuracy(model, cam_val)
        if name == "V14_extra_data_train":
            ext_c, ext_n = None, len(external_val)  # external is training data here
        else:
            ext_c, ext_n = accuracy(model, external_val)
        if seed == 0:
            torch.save(model.state_dict(), f"task4_models/{name}_seed0.pt")
        result = {
            "seed": seed,
            "final_loss": loss_sum / len(loader),
            "epoch_time_s": round(epoch_time, 2),
            "clean_correct": clean_c, "clean_total": clean_n,
            "cam_correct": cam_c, "cam_total": cam_n,
            "ext_correct": ext_c, "ext_total": ext_n,
        }
        results.append(result)
        PROGRESS.setdefault(name, {})[str(seed)] = result
        save_progress()
        print(f"  {name} seed {seed}: clean {clean_c}/{clean_n}, cam {cam_c}/{cam_n}, "
              f"ext {ext_c}/{ext_n}, {epoch_time:.1f}s/epoch")
    return results


# ---- the variants --------------------------------------------------------------

variants = {
    "V1_leafnet_base":        dict(model=LeafNet,    epochs=15, lr=1e-3),
    "V2_leafnet_aug":         dict(model=LeafNet,    epochs=15, lr=1e-3, aug=True),
    "V3_leafnet_bn":          dict(model=LeafNetBN,  epochs=15, lr=1e-3),
    "V4_leafnet_bn_aug_15ep": dict(model=LeafNetBN,  epochs=15, lr=1e-3, aug=True),
    "V5_more_epochs":         dict(model=LeafNetBN,  epochs=30, lr=1e-3, aug=True),
    "V6_cosine_lr":           dict(model=LeafNetBN,  epochs=15, lr=1e-3, aug=True, cosine=True),
    "V7_lr_3e4":              dict(model=LeafNetBN,  epochs=15, lr=3e-4, aug=True),
    "V8_weight_decay":        dict(model=LeafNetBN,  epochs=15, lr=1e-3, aug=True, weight_decay=1e-4),
    "V9_dropout":             dict(model=LeafNetBNDrop, epochs=15, lr=1e-3, aug=True),
    "V10_stronger_geom_aug":  dict(model=LeafNetBN,  epochs=15, lr=1e-3, aug="strong"),
    "V10b_color_jitter":      dict(model=LeafNetBN,  epochs=15, lr=1e-3, aug="color"),
}

V11 = dict(model=LeafNetBN, epochs=15, lr=1e-3, aug=True)  # starts from V4, patched below
chosen_changes = []


def dataset_for(cfg):
    if cfg.get("aug") is True:
        return LeafDataset("data/leaves/train", aug_transform)
    if cfg.get("aug") == "strong":
        return LeafDataset("data/leaves/train", aug_strong_transform)
    if cfg.get("aug") == "color":
        return LeafDataset("data/leaves/train", aug_color_transform)
    if cfg.get("aug") in ("strong+color", "color+strong"):
        # both V10 and V10b survived: stronger crop scale AND ColorJitter
        strong_color_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return LeafDataset("data/leaves/train", strong_color_transform)
    return train_ds_plain


# Time check: run V4 seed 0 for one epoch, and one MobileNet head-only epoch,
# then estimate the total before committing to everything.
print("\nTime check (1-epoch probes)...")
probe_cfg = variants["V4_leafnet_bn_aug_15ep"]
probe_ds = dataset_for(probe_cfg)
torch.manual_seed(0)
probe_model = probe_cfg["model"]().to(device)
probe_opt = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
probe_loader = DataLoader(probe_ds, batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
t0 = time.time()
probe_model.train()
for images, labels in probe_loader:
    probe_opt.zero_grad()
    nn.CrossEntropyLoss()(probe_model(images), labels).backward()
    probe_opt.step()
leaf_epoch_s = time.time() - t0

probe_mob = make_mobilenet().to(device)
for p in probe_mob.features.parameters():
    p.requires_grad_(False)
mob_opt = torch.optim.Adam(probe_mob.classifier[3].parameters(), lr=1e-3)
mob_ds = LeafDataset("data/leaves/train", aug_transform)
mob_loader = DataLoader(mob_ds, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
t0 = time.time()
probe_mob.train(); probe_mob.features.eval()
for images, labels in mob_loader:
    mob_opt.zero_grad()
    nn.CrossEntropyLoss()(probe_mob(images), labels).backward()
    mob_opt.step()
mob_head_epoch_s = time.time() - t0
mob_full_epoch_s = mob_head_epoch_s * 2.5  # rough: adds the backward through the backbone

n_leaf_epochs = sum(cfg["epochs"] for cfg in variants.values()) * len(SEEDS)  # V1..V10b
n_leaf_epochs += V11["epochs"] * len(SEEDS)
v14_scale = (244 + 820) / 244  # V14 sees ~4.4x the images per epoch
estimated_s = (n_leaf_epochs * leaf_epoch_s
               + V11.get("epochs", 15) * len(SEEDS) * 0  # V11 already counted above
               + 15 * len(SEEDS) * leaf_epoch_s * v14_scale   # V14
               + 15 * len(SEEDS) * mob_head_epoch_s            # V12
               + 15 * len(SEEDS) * mob_full_epoch_s)           # V13
print(f"1 LeafNet(+BN) epoch: {leaf_epoch_s:.1f}s, 1 MobileNet head-only epoch: {mob_head_epoch_s:.1f}s")
print(f"estimated total for all variants x {len(SEEDS)} seeds: {estimated_s / 60:.1f} minutes")
if estimated_s > TIME_BUDGET_S:
    print(f"WARNING: the estimate exceeds the ~1 hour budget ({TIME_BUDGET_S}s). "
          "Stopping before running everything, as instructed.")
    sys.exit(1)

results = {}
for name, cfg in variants.items():
    print(f"\n{name}")
    results[name] = train_variant(
        name, cfg["model"], dataset_for(cfg), cfg["epochs"], cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0), cosine=cfg.get("cosine", False))

# ---- which single changes genuinely improved over V4? ---------------------------
# Criterion from the task: mean improvement over V4 > std, AND at least 2 extra
# correct images on cam_val or external_val (the 62 clean images alone are noise).
v4 = results["V4_leafnet_bn_aug_15ep"]

def delta_stats(name, key):
    d = [results[name][i][key] - v4[i][key] for i in range(len(SEEDS))]
    return statistics.mean(d), statistics.stdev(d)

print("\nSingle changes vs V4 (correct-image deltas, mean +- std over seeds):")
decision_lines = []
for name, setting in [("V5_more_epochs", "epochs=30"), ("V6_cosine_lr", "cosine"),
                      ("V7_lr_3e4", "lr=3e-4"), ("V8_weight_decay", "wd=1e-4"),
                      ("V9_dropout", "dropout0.3"), ("V10_stronger_geom_aug", "aug scale(0.4,1.0)"),
                      ("V10b_color_jitter", "colorjitter")]:
    cam_m, cam_s = delta_stats(name, "cam_correct")
    ext_m, ext_s = delta_stats(name, "ext_correct")
    ok_cam = cam_m > cam_s and cam_m >= 2
    ok_ext = ext_m > ext_s and ext_m >= 2
    keep = ok_cam or ok_ext
    if keep:
        chosen_changes.append(setting)
    decision_lines.append(
        f"{name}: cam {cam_m:+.2f}+-{cam_s:.2f}, ext {ext_m:+.2f}+-{ext_s:.2f} -> "
        f"{'KEPT' if keep else 'dropped'}")
    print("  " + decision_lines[-1])
print("V11 combines:", chosen_changes if chosen_changes else "(nothing beat V4 -> V11 = V4 config)")

for setting in chosen_changes:
    if setting == "epochs=30":
        V11["epochs"] = 30
    elif setting == "cosine":
        V11["cosine"] = True
    elif setting == "lr=3e-4":
        V11["lr"] = 3e-4
    elif setting == "wd=1e-4":
        V11["weight_decay"] = 1e-4
    elif setting == "dropout0.3":
        V11["model"] = LeafNetBNDrop
    elif setting == "aug scale(0.4,1.0)":
        V11["aug"] = "strong"
    elif setting == "colorjitter":
        V11["aug"] = "color" if V11.get("aug") is True else V11["aug"] + "+color"

print("\nV11_best_leafnet_combo")
results["V11_best_leafnet_combo"] = train_variant(
    "V11_best_leafnet_combo", V11["model"], dataset_for(V11), V11["epochs"], V11["lr"],
    weight_decay=V11.get("weight_decay", 0.0), cosine=V11.get("cosine", False))

print("\nV12_mobilenet_head_only")
results["V12_mobilenet_head_only"] = train_variant(
    "V12_mobilenet_head_only", make_mobilenet, LeafDataset("data/leaves/train", aug_transform),
    15, 1e-3, head_only=True)

print("\nV13_mobilenet_finetune_all")
results["V13_mobilenet_finetune_all"] = train_variant(
    "V13_mobilenet_finetune_all", make_mobilenet, LeafDataset("data/leaves/train", aug_transform),
    15, 3e-4, finetune_all=True)

print("\nV14_extra_data_train")
v14_ds = ConcatDataset([
    LeafDataset("data/leaves/train", aug_transform),
    LeafDataset("external_data/leaves/train", aug_transform),
    LeafDataset("external_data/leaves/val", aug_transform),
])
results["V14_extra_data_train"] = train_variant(
    "V14_extra_data_train", LeafNetBN, v14_ds, 15, 1e-3)

# ---- summary ---------------------------------------------------------------------

def summarize(rs):
    out = {}
    for key, label in [("clean", "clean_val"), ("cam", "cam_val"), ("ext", "external_val")]:
        vals = [100 * r[f"{key}_correct"] / r[f"{key}_total"] for r in rs
                if r[f"{key}_correct"] is not None]
        out[label] = None if not vals else (statistics.mean(vals),
                                            statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return out

lines = [f"Task 4 experiments - {time.strftime('%Y-%m-%d %H:%M')} - device {device}",
         f"clean_val 62 imgs | cam_val 310 imgs (seed 999) | external_val {len(external_val)} imgs",
         f"probes: leafnet epoch {leaf_epoch_s:.1f}s, mobilenet head epoch {mob_head_epoch_s:.1f}s",
         f"estimate {estimated_s / 60:.1f} min (budget 60)",
         "", "V11 combined changes:", *(f"  {c}" for c in chosen_changes), ""]
table = []
for name, rs in results.items():
    s = summarize(rs)
    def fmt(v):
        return "   n/a    " if v is None else f"{v[0]:5.1f}+-{v[1]:4.1f}"
    row = f"{name:26s} clean {fmt(s['clean_val'])}  cam {fmt(s['cam_val'])}  ext {fmt(s['external_val'])}"
    table.append(row)

suspicious = []
for name, rs in results.items():
    for r in rs:
        if r["clean_correct"] == r["clean_total"]:
            suspicious.append(f"{name} seed {r['seed']}: 100% clean_val (suspicious: check leakage)")
        if any(v != v for v in [r["final_loss"]] if v is not None):
            suspicious.append(f"{name} seed {r['seed']}: NaN loss")
ext_names = {os.path.basename(p) for p, _ in external_val.datasets[0].samples + external_val.datasets[1].samples}
own_train_names = {os.path.basename(p) for p, _ in train_ds_plain.samples}
overlap = ext_names & own_train_names
if overlap:
    suspicious.append(f"{len(overlap)} filenames shared between own train and the external set "
                      f"(same photo? would inflate external_val): {sorted(overlap)[:5]}...")
if suspicious:
    table.append("")
    table.append("SUSPICIOUS / NOTES:")
    table.extend("  " + s for s in suspicious)

lines.extend(table)
lines.extend(["", "Per-seed raw numbers:",
              json.dumps(results, indent=1, default=str)])
summary_text = "\n".join(lines)
print("\n" + summary_text)
with open("task4_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)
with open("task4_results.json", "w", encoding="utf-8") as f:
    json.dump({"probes": {"leafnet_epoch_s": leaf_epoch_s, "mobilenet_head_epoch_s": mob_head_epoch_s},
               "chosen_changes": chosen_changes, "results": results}, f, indent=1, default=str)
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)  # run complete; the resume file is no longer needed
print("\nwrote task4_results.json and task4_summary.txt")
