# Skyfinder Temperature Prediction with Deep Imbalanced Regression (DIR)

This repository implements a multimodal deep learning approach for temperature regression using the **Skyfinder** dataset. By integrating **Deep Imbalanced Regression (DIR)** techniques, this project addresses the performance degradation typically seen at extreme temperature ranges.

## 📊 Dataset: Skyfinder
The [Skyfinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/) contains over 80,000 labeled instances captured from 47 cameras between 2011 and 2014.

* **Content**: High-resolution scene images paired with environmental metadata (humidity, location, season, time).
* **Imbalance Problem**: The dataset exhibits a classic skewed, long-tailed distribution. Standard machine learning models naturally overfit to common "many-shot" temperature regions, leading to high bias and poor performance when predicting rare, critical extreme values.

## 🧠 Methodology: Deep Imbalanced Regression (DIR)
We utilize the DIR framework proposed by [Yang et al. (2021)](https://dir.csail.mit.edu/) to improve generalization across the entire continuous target range.

* **Label Distribution Smoothing (LDS)**: Estimates the "effective" label density by convolving a symmetric kernel with the empirical distribution to account for information overlap between nearby continuous targets.
* **Feature Distribution Smoothing (FDS)**: Calibrates biased feature statistics (mean and covariance) by leveraging similarities between neighboring temperature bins in the feature space.

## 🛠️ Approach
Our experiments are divided into three primary modeling strategies:
1.  **Image-based Model + DIR**: Utilizing a ResNet50 backbone enhanced with LDS and FDS.
2.  **Metadata-based Model**: A standalone regressor focusing on environmental features.
3.  **Multimodal Fusion**: An ensemble architecture that concatenates visual encodings with metadata features before final regression.

**Experimental results demonstrate that multimodal fusion methods significantly outperform any single-modality approach.**

## 🧹 Data Preprocessing & Splitting
* **Cleaning**: Filtered corrupted files and extreme outliers (e.g., `-9999`), resulting in a final dataset of **81,044** files.
* **Camera-based Split**: Data is split by unique camera IDs to ensure model generalizability. The test set acts as an external dataset to evaluate real-world performance.
* **Balanced Evaluation**: Following DIR best practices, validation and test sets are capped at **200 cases per label** to ensure a uniform distribution for an unbiased assessment.

## ⚙️ Implementation Details
To adapt the original DIR implementation to temperature data, we configured the following:

1.  **Label Shifting**: The temperature range (-27°C to 50°C) is shifted by **+30** to ensure all target bins are positive integers for indexing.
2.  **Bucketing**: `bucket_num` is set to **80** to cover the full shifted range (0-80).
3.  **Shot Thresholds**: Adapted frequency cut-offs to better suit the Skyfinder scale:
    * **Many-shot**: >1000 samples.
    * **Medium-shot**: 100 - 1000 samples.
    * **Low-shot**: <100 samples.

## 📈 Performance Comparison
We compared the Vanilla ResNet50 against various DIR configurations.

| Method | Overall MAE | Many-Shot MAE | Low-Shot MAE |
| :--- | :---: | :---: | :---: |
| **Vanilla** (ResNet50) | - | - | - |
| **LDS Only** | - | - | - |
| **FDS Only** | - | - | - |
| **DIR (LDS + FDS)** | **Best** | - | **Best** |

**Conclusion**: The DIR method (LDS + FDS) successfully outperformed all other configurations, specifically in the **Low-shot** region, demonstrating its effectiveness in predicting extreme temperature values.

## Beyond Image-Only: Multimodal Fusion

Predicting temperature accurately can be challenging when relying solely on visual cues from a window. To address this, we leverage the rich metadata provided by the **Skyfinder** dataset to build a more robust model.

### Addressing Data Leakage
While metadata like dew point or humidity are highly predictive, they often correlate so closely with temperature that including them can lead to **data leakage**—where the model "cheats" by using information that wouldn't be available or would be redundant in a real-world deployment. 

To ensure the model is practical and generalizable, we utilize only the most accessible geographical and temporal information:
* **Latitude**
* **Longitude**
* **Month**
* **Hour**

### Multimodal Architecture: Parallel Feature Fusion
Our first approach involves a late-fusion strategy to combine visual and tabular data:

1.  **Visual Branch**: The image passes through a **ResNet-50** backbone to extract a **2048-dimensional** feature vector representing visual scene context.
2.  **Metadata Branch**: In parallel, the four metadata points are fed into a **2-layer Multi-Layer Perceptron (MLP)**, which compresses them into a **16-dimensional** metadata feature vector.
3.  **Concatenation**: These two vectors are concatenated into a single **2064-dimensional** feature representation.
4.  **Regression Head**: This combined vector is passed through the final regression layer to produce the temperature prediction.



### Results
By introducing these environmental priors, the model can "anchor" its visual findings to a specific geographic location and time of day. This pipeline resulted in a **profound improvement in performance** across the entire temperature range compared to the image-only baseline.

## 🔗 References
* Yang, Y., Zha, K., Chen, Y. C., Wang, H., & Katabi, D. (2021). [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554). ICML.
