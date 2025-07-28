# COVID-19 GLCM Chest X-Ray Classifier

This project develops a machine learning pipeline to classify chest X-ray images into three categories: COVID-19, Pneumonia, and Normal. It uses Gray Level Co-occurrence Matrix (GLCM) features extracted from grayscale images and applies traditional classifiers to predict conditions based on texture patterns in the X-rays.

---

## Key Features

- Dataset of 603 X-ray images sourced from Mendeley, aggregated from 3 public repositories
- Preprocessing pipeline with **Roboflow object detection API** for cleaning ECG leads and text
- Dual feature extraction:
  - Flattened pixel arrays
  - 36 GLCM texture features across multiple distances and angles
- Classification with **Logistic Regression**, **Random Forest**, **SVC**, and **k-NN**
- Feature selection using **Permutation Importance** on GLCM features
- Model optimization using **GridSearchCV**
- Evaluation with ROC-AUC (One-vs-Rest), precision, recall, F1-score, confusion matrices

---

## Example Predictions

- Input: Chest X-ray of COVID-19 → Predicted: COVID-19  
- Input: Chest X-ray of Pneumonia → Predicted: Pneumonia  
- Input: Chest X-ray of Healthy lung → Predicted: Normal  

Best performance achieved by SVC model using GLCM features with low-importance features removed.

---

## Project Structure

```
covid-xray-classifier/
├── dataset/ # Organized image folders for COVID-19, Pneumonia, Normal
├── glcm_features.npy # Extracted features (optional: cached)
├── models/ # Saved trained models (optional)
├── main.ipynb # Main notebook (feature extraction, training, evaluation)
├── README.md
└── .gitignore
```

---

## Technologies Used

- **scikit-learn** – classification models, feature selection, hyperparameter tuning
- **OpenCV** – image manipulation
- **scikit-image** – GLCM texture feature extraction
- **Roboflow API** – ECG/text artifact removal
- **NumPy**, **Pandas** – data processing
- **Matplotlib**, **Seaborn** – data visualization

---

## Evaluation Summary

| Model | Feature Set | Accuracy | Macro F1 | Macro ROC-AUC |
|-------|-------------|----------|----------|----------------|
| Logistic Regression | Flattened Pixels | 88.40% | 0.8825 | 0.9733 |
| SVC | GLCM Full | 87.85% | 0.8765 | 0.9586 |
| SVC | GLCM Selected | **88.95%** | **0.8876** | **0.9546** |

- **Best overall model**: SVC using selected GLCM features
- GLCM-based models outperform raw pixel models on Pneumonia classification

---

## Limitations

- Dataset is relatively small (603 images)
- ECG/text black-box masking may introduce minor artifacts
- No deep learning baseline for comparison (e.g. CNN)
- Generalization to unseen datasets may require further validation

---

## Future Work

- Expand dataset and ensure higher diversity
- Compare against CNN and deep learning-based classifiers
- Combine GLCM features with CNN outputs for hybrid modeling
- Refine artifact removal using custom segmentation methods

---

## Notes

This study demonstrates the feasibility of GLCM-based texture features for medical image classification using classical machine learning techniques. The code and models are intended for academic and experimental use only.
