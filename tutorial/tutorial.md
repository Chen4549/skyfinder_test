# Evaluation Tutorial

This repository contains the evaluation scripts for our work. The evaluation is divided into three main components:
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

## 2. Model Checkpoints
1. Download the pre-trained weights for the visual backbone here: [Trained Visual Backbone (ResNet50): ](https://drive.google.com/file/d/1V1J7Kl0pYi-xetWGhLBhnC0gZFyYhwk1/view?usp=drive_link)
2. Place the downloaded file in the following location: `./imbalanced-regression/Skyfinder-dir/checkpoint/ckpt.best.pth.tar`

## 3. Running Evaluation

Once the data and checkpoints are in place, you can proceed with the evaluation scripts.

**Evaluate TabPFN (Metadata Only)**

To evaluate the performance of TabPFN using metadata only, run: `python tabPFN/tab_regression.py`

**Evaluate ResNet+DIR and Multimodal TabPFN**

To get the performance results for both the ResNet+DIR model and the Multimodal TabPFN framework, run: `python ./fusion/concat_pfn.py`
