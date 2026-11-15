# Task 4 Report - Improving the Leaf Classifier

## 1. Setup and validation sets

All experiments are in task4_experiments.py and the verification analysis is in task4_inspect_and_export.py. Both scripts are included in the repository. They start from my Task 3 pipeline (same LeafDataset, LeafNet, LeafNetBN, and Adam training loop with batch size 32). Before running everything on CPU, a 1-epoch check measured ~0.6 s per epoch for LeafNetBN (probe 1.3 s) and ~0.32 s per epoch for frozen MobileNetV3-Small (probe 0.7 s), with an estimated total runtime of 18.3 minutes.

Our group dataset in data/leaves has 244 training images (68 apple, 75 cherry, 41 chestnut, 60 maple) and 62 validation images (17 apple, 19 cherry, 11 chestnut, 15 maple). Because 1 validation image is 1/62 = 1.61% accuracy, a difference of 2 or 3 images on clean_val is within random noise. To get better estimates, every variant was run with 3 seeds (0, 1, 2) and evaluated on three sets:

- clean_val (62 images): the unperturbed validation split in data/leaves/val.
- cam_val (310 images): 5 perturbed copies per validation image generated once with seed 999 using RandomResizedCrop(96, scale=(0.5, 1.0)), RandomRotation(25), and ColorJitter(0.3, 0.3, 0.2, 0.05). Because the 5 copies come from the same 62 source images, they are correlated, and cam_val uses the same ColorJitter parameters as V10b, which favors V10b on this synthetic set.
- external_val (820 images): last year's dataset from ex08_leaves_images.zip (700 train + 120 val, balanced with 205 images per class), used strictly for evaluation in V1 to V13. I checked that both datasets have the exact same class order ['apple', 'cherry', 'chestnut', 'maple'] and 96x96 RGB images. Although 244 train and 62 val filenames overlap (like apple_000.jpg), comparing MD5 hashes across all 488 shared filename pairs showed 0 byte-identical files (near-duplicate photos of the same physical leaf were not tested).

## 2. Results across 3 seeds (0, 1, 2)

```text
Format: Variant | clean_val (out of 62) | cam_val (out of 310) | external_val (out of 820) | mean final train loss
(Note: train loss is not comparable between runs with and without augmentation; it is only listed as an indicator of overfitting within the same augmentation setup.)

Task 3 baselines (15 epochs, Adam lr=1e-3):
V1_leafnet_base:            71.0 +- 1.6% (44.0/62) | 42.5 +- 3.8% (131.7/310) | 37.4 +- 4.0% (307.0/820) | loss 0.244
V2_leafnet_aug:             69.4 +- 4.3% (43.0/62) | 63.0 +- 1.8% (195.3/310) | 42.9 +- 8.1% (351.7/820) | loss 0.704
V3_leafnet_bn:              74.2 +- 3.2% (46.0/62) | 49.4 +- 1.5% (153.0/310) | 44.6 +- 1.5% (366.0/820) | loss 0.020
V4_leafnet_bn_aug_15ep:     73.7 +- 4.9% (45.7/62) | 55.6 +- 2.7% (172.3/310) | 45.4 +- 4.9% (372.0/820) | loss 0.502

Single changes tested one at a time against V4:
V5_more_epochs (30 ep):     77.4 +- 4.3% (48.0/62) | 62.2 +- 3.9% (192.7/310) | 41.0 +- 2.4% (336.3/820) | loss 0.396
V6_cosine_lr:               71.5 +- 0.9% (44.3/62) | 58.8 +- 4.4% (182.3/310) | 46.2 +- 2.4% (379.0/820) | loss 0.490
V7_lr_3e4:                  69.4 +- 5.8% (43.0/62) | 56.2 +- 2.3% (174.3/310) | 45.3 +- 4.4% (371.3/820) | loss 0.504
V8_weight_decay (1e-4):     72.0 +- 6.5% (44.7/62) | 56.2 +- 4.4% (174.3/310) | 42.5 +- 6.9% (348.3/820) | loss 0.502
V9_dropout (0.3 in fc):     63.4 +- 8.0% (39.3/62) | 53.8 +- 5.5% (166.7/310) | 44.1 +- 7.2% (362.0/820) | loss 0.585
V10_stronger_geom_aug:      71.0 +- 1.6% (44.0/62) | 57.6 +- 2.4% (178.7/310) | 44.8 +- 3.4% (367.0/820) | loss 0.593
V10b_color_jitter:          74.7 +- 5.7% (46.3/62) | 67.4 +- 1.2% (209.0/310) | 46.3 +- 2.4% (379.3/820) | loss 0.679

Combination, pretrained MobileNetV3-Small, and extra training data:
V11_best_leafnet_combo:     73.1 +- 6.5% (45.3/62) | 70.8 +- 2.0% (219.3/310) | 45.8 +- 2.4% (375.3/820) | loss 0.529
V12_mobilenet_head_only:    83.3 +- 2.5% (51.7/62) | 84.3 +- 1.2% (261.3/310) | 46.4 +- 2.6% (380.3/820) | loss 0.180
V13_mobilenet_finetune_all: 81.2 +- 8.9% (50.3/62) | 77.4 +- 1.7% (240.0/310) | 45.7 +- 1.4% (374.3/820) | loss 0.154
V14_extra_data_train:       65.1 +- 1.9% (40.3/62) | 58.4 +- 0.9% (181.0/310) | n/a (used in training)   | loss 0.475

```

## 3. What helped, what was noise, and why

1. Task 3 baselines (V1 to V4):
On clean_val, V1 to V4 are all between 43.0 and 46.0 correct images out of 62. Without augmentation (V1, V3), models overfit to the clean images (V3 train loss is 0.020) and get only 42.5% and 49.4% on cam_val. Adding crop/flip/rotation augmentation in V2 and V4 raises cam_val to 63.0% and 55.6%. I used V4 as the baseline for testing individual changes.
2. Single changes against V4 (V5 to V10b):
In the script, a change was selected for V11 if its paired gain over V4 had mean > std and at least 2 images on cam_val or external_val (on 310 and 820 images, 2 images is <0.7%, so in practice the mean > std rule decided, and with only 3 seeds the std estimate itself is noisy):

* V10b (ColorJitter): gained +36.67 +- 10.50 images on cam_val (67.4% vs 55.6%). Part of this gain on cam_val is because cam_val was generated with the same ColorJitter transform. On external_val, the change was +7.33 +- 55.87 images (46.3% vs 45.4%), which is noise. So ColorJitter helps against synthetic color shifts, while whether it helps on a real phone camera requires testing on the phone.
* V5 (30 epochs): gained +20.33 +- 13.80 images on cam_val (62.2%) and reached 77.4% on clean_val. On external_val it dropped by -35.67 +- 21.03 images (41.0% vs 45.4% in V4, where V4 std is 4.9%), which is within noise or maybe slight overfitting to our dataset.
* V6 (CosineAnnealingLR), V7 (lr=3e-4), V8 (weight_decay=1e-4), and V10 (stronger crop scale 0.4-1.0): all had std larger than their mean change and were dropped.
* V9 (Dropout 0.3 in fc): hurt performance on clean_val (63.4 +- 8.0%) and cam_val (53.8 +- 5.5%), likely because 15 epochs with augmentation and dropout on 244 images is too short to converge well.

3. Combination V11:
Combining 30 epochs and ColorJitter gave 70.8 +- 2.0% on cam_val and 45.8 +- 2.4% on external_val. Because V11 was built from the changes that scored best on cam_val and external_val, its evaluation on those same sets is not independent and is slightly optimistic.
4. Pretrained MobileNetV3-Small (V12 vs V13):

* V12 (frozen backbone in eval() mode, training only the linear head) was the best model on clean_val (83.3 +- 2.5%, 51.7/62) and cam_val (84.3 +- 1.2%, 261.3/310).
* V13 (fine-tuning all layers at lr=3e-4) was clearly worse than V12 on cam_val across all seeds (77.4 +- 1.7% vs 84.3 +- 1.2%). On clean_val (81.2 +- 8.9% vs 83.3 +- 2.5%) and external_val (45.7% vs 46.4%) the two cannot be distinguished within variance, although V13 was less stable across seeds on clean_val (seed 0 got 44/62 while seeds 1-2 got 53-54/62).

5. Confusion matrices on external_val (seed 0) and V14:
As computed by task4_inspect_and_export.py, seed 0 confusion matrices show distinct class biases under domain shift:

* On clean_val (n=62), V12 seed 0 gets 83.9% (52/62).


* On external_val (n=820, 205 per class), V12 seed 0 drops to 44.5% because V12 strongly over-predicts apple (472 of 820 predictions, about 57%, while cherry recall is only 79/205).


* V4 seed 0 gets 50.5% on external_val, but V4 is biased towards chestnut (398 of 820 predictions, about 49%).
This shows that under domain shift between our photos and last year's photos, each model develops its own directional bias. In V14, training on our 244 images plus the 820 external images lowered clean_val to 65.1 +- 1.9%, maybe because the larger external set shifts the decision boundary away from our group's validation photos.



## 4. Preliminary model choice for the phone and ONNX export

My preliminary choice for phone deployment is V12 (MobileNetV3-Small head-only, seed 0 exported to task4_best_v12.onnx, 6.1 MB, input image [1,3,96,96], output logits [1,4], opset 18). However, this recommendation has important caveats:

* V12's advantage is only proven on clean_val (83.3%) and cam_val (84.3%), which come from the same photo session as our training data. On external_val, all models perform similarly around 41-46% (V12 has a mean of 46.4%, V11 has 45.8%, and V4 has a mean of 45.4%), so better performance under new camera lighting is not guaranteed.
* On check_image.jpg (where the Task 2 model gave 98.34% apple), V12 seed 0 is almost a coin flip between cherry (55.37%) and apple (44.62%), matching its apple/cherry confusion in the clean_val confusion matrix (PyTorch and ONNX outputs agree to max diff 4.8e-6).


* Because V12 uses the same MobileNetV3-Small architecture as the Task 2 model, I expect a similar inference time (~9-16 ms) in the browser. Note that the 4 screenshot files in the repo are full phone UI screenshots rather than raw leaf photos, so running center-crop inference on them is not a valid leaf test.
* I kept the original leaf_model.onnx in the repo so the deployed Task 2 page and REVIEW.md stay unchanged, and saved V12 seed 0 as task4_best_v12.onnx.


* The model has not yet been tested in the live course camera app. Therefore, this deployment choice and performance under live camera conditions remain preliminary.

