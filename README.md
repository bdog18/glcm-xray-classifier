# COVID-19 GLCM Chest X-Ray Classifier

This project develops a machine learning pipeline to classify chest X-ray images into three categories: COVID-19, Pneumonia, and Normal. It uses Gray Level Co-occurrence Matrix (GLCM) features extracted from grayscale images and applies traditional classifiers to predict conditions based on texture patterns in the X-rays.

---

## Key Features

- End-to-end workflow: image preprocessing → GLCM feature extraction → model training → evaluation
- Classifies X-rays into COVID-19, Pneumonia, or Normal categories
- Extracts 14 GLCM texture features including contrast, correlation, and entropy
- Trains multiple classifiers: Logistic Regression, Random Forest, k-NN, SVM, and XGBoost
- Model performance comparison using accuracy, classification report, and confusion matrix
- Misclassification analysis with confusion heatmaps
- Robust image pipeline using OpenCV and scikit-image

---

## Example Predictions

- Input: Chest X-ray of COVID-19 → Predicted: COVID-19  
- Input: Chest X-ray of Pneumonia → Predicted: Pneumonia  
- Input: Chest X-ray of Healthy lung → Predicted: Normal  

Confusions occur mostly between Pneumonia and COVID-19 due to overlapping visual patterns.

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

- scikit-learn (ML models, evaluation)
- OpenCV (image processing)
- scikit-image (GLCM feature extraction)
- NumPy, Pandas
- Matplotlib, Seaborn

---

## Evaluation Metrics

- Best model accuracy: ~88.95% (SVC Model using GLCM features based on permutation importance)
- Confusion Matrix: Strong separation between Normal and disease classes
- Classification Report: Balanced precision and recall; COVID-19 class slightly underperforms due to data imbalance

---

## Limitations

- Small dataset size may limit generalizability
- GLCM features are sensitive to image resolution and lighting
- Performance may degrade on images from different sources or clinical settings

---

## Future Work

- Apply deep learning (CNNs) for improved spatial feature learning
- Explore data augmentation to increase robustness
- Integrate additional feature sets (e.g., wavelet transforms, HOG)
- Build a user interface for medical staff to upload and classify X-rays
- Add cross-validation for better model generalization

---

## Notes

This project is intended for academic and research use only. The model is **not** approved for clinical diagnosis. Feedback and contributions are welcome to improve performance and usability.
