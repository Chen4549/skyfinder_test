# Evaluation Tutorial

This repository contains the evaluation scripts for my work. The evaluation is divided into three main components:
1. Performance of **ResNet50 + DIR** on the test set.
2. Performance of **TabPFN** using metadata only.
3. Performance of the **TabPFN Multimodal Framework**.

---

## 1. Data Preparation

First, download the dataset from the [Skyfinder website](https://cs.valdosta.edu/~rpmihail/skyfinder/). 

Extract the images into the following directory:  
`./imbalanced-regression/Skyfinder-dir/data/`

Ensure the directory structure follows this format:
```text
/path/to/workingDIR/skyfinder_test/imbalanced-regression/Skyfinder-dir/data/
├── 123 (CamID)/
│   ├── image_example1.png
│   ├── image_example2.png
│   └── ...
├── 456 (CamID)/
│   ├── image_example1.png
│   ├── image_example2.png
│   └── ...
└── ...
```

---

## 2. Model Checkpoints
1. Download the pre-trained weights for the visual backbone here: [Trained Visual Backbone (ResNet50): ](https://drive.google.com/file/d/1V1J7Kl0pYi-xetWGhLBhnC0gZFyYhwk1/view?usp=drive_link)
2. Place the downloaded file in the following location: `./imbalanced-regression/Skyfinder-dir/checkpoint/ckpt.best.pth.tar`

---

## 3. Conda Env set up:

Conda env is saved to a .yml file at [Env yaml](./environment.yml)

`conda env create -f skyfinder.yml`

then

`conda activate skyfinder-env`

---

## 4. Running Evaluation

Once the data and checkpoints are in place, you can proceed with the evaluation scripts.

**Evaluate TabPFN (Metadata Only)**

To evaluate the performance of TabPFN using metadata only, run: `python tabPFN/tab_regression.py`

script is under [TabPFN metadata only script](./tabPFN/tab_regression.py)

Output is following:

```bash
(skyfinder-env) [hc4549@a100-4022 skyfinder_test]$ python ./tabPFN/tab_regression.py
TabPFN Input features: ['Latitude', 'Longitude', 'Month', 'Hour']
Target variable name: label

--- TabPFN EVALUATION RESULTS ---
Overall: MSE: 39.4655 | MAE: 5.1631 | Count: 5117
Many: MSE: 35.3999 | MAE: 4.7426 | Count: 3903
Medium: MSE: 47.7638 | MAE: 6.0816 | Count: 800
Low: MSE: 61.7589 | MAE: 7.3519 | Count: 414
```

**Evaluate ResNet+DIR and Multimodal TabPFN**

To get the performance results for both the ResNet+DIR model and the Multimodal TabPFN framework, run: `python ./fusion/concat_pfn.py`

script is under [ResNet+DIR and Multimodal TabPFN script](./fusion/concat_pfn.py)

Output is following:

```bash
(skyfinder-env) [hc4549@a100-4022 skyfinder_test]$ python ./fusion/concat_pfn.py
=====> Preparing data...
File (.csv): tabpfn.csv
Training data size: 10000
Test data size: 5117

=====> Building model...
===> Checkpoint ./imbalanced-regression/Skyfinder-dir/checkpoint/ckpt.best.pth.tar loaded (epoch [50]), testing...

--------------------------------------------------
===>Train set performance: 
 * Overall: MSE 24.868  L1 3.794        G-Mean 2.405
 * Many: MSE 23.920     L1 3.729        G-Mean 2.366
 * Median: MSE 29.980   L1 4.179        G-Mean 2.651
 * Low: MSE 42.690      L1 4.679        G-Mean 2.925
===> Extracting 2048-d visual features with mapping...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:13<00:00,  2.99it/s]

--------------------------------------------------
===>Test set performance: 
 * Overall: MSE 118.860 L1 8.216        G-Mean 5.104
 * Many: MSE 68.663     L1 6.227        G-Mean 3.904
 * Median: MSE 203.453  L1 12.053       G-Mean 9.676
 * Low: MSE 428.632     L1 19.558       G-Mean 18.565
===> Extracting 2048-d visual features with mapping...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:06<00:00,  3.08it/s]

--------------------------------------------------
Applying PCA to compress feature to: 256
PCA compress completed.

--------------------------------------------------
TabPFN input Features:
  Metadata: ['Latitude', 'Longitude', 'Month', 'Hour']
  PCA Components: 0, 1, ..., 255 (256 total)
Target variable name: label

--- FINAL(RESNET50+TABPFN) EVALUATION RESULTS ---
Overall: MSE: 42.6316 | MAE: 5.1915 | Count: 5117
Many: MSE: 43.8716 | MAE: 5.1540 | Count: 3903
Medium: MSE: 38.9378 | MAE: 5.1816 | Count: 800
Low: MSE: 38.0791 | MAE: 5.5640 | Count: 414
```

ResNet+DIR test set result is under:
```bash
===>Test set performance: 
 * Overall: MSE 118.860 L1 8.216        G-Mean 5.104
 * Many: MSE 68.663     L1 6.227        G-Mean 3.904
 * Median: MSE 203.453  L1 12.053       G-Mean 9.676
 * Low: MSE 428.632     L1 19.558       G-Mean 18.565
```

Multimodal TabPFN result is under:
```bash
--- FINAL(RESNET50+TABPFN) EVALUATION RESULTS ---
Overall: MSE: 42.6316 | MAE: 5.1915 | Count: 5117
Many: MSE: 43.8716 | MAE: 5.1540 | Count: 3903
Medium: MSE: 38.9378 | MAE: 5.1816 | Count: 800
Low: MSE: 38.0791 | MAE: 5.5640 | Count: 414
```

The output also contain information on which data and metadata is used, and how many image features were used.
