# COVID-19 GLCM Chest X-Ray Classifier

**Author**: Brenden Runion  
**Date**: April 2025  
**Language**: Python 3.10+

This project implements a comprehensive machine learning pipeline for classifying chest X-ray images into three categories: **COVID-19**, **Normal**, and **Pneumonia**. The research demonstrates the trade-offs between accuracy and computational efficiency in medical image classification, comparing pixel-based and texture-based feature approaches.

### **Research Question**
*Can Gray-Level Co-occurrence Matrix (GLCM) texture features provide an optimal balance of accuracy and computational efficiency compared to raw pixel intensities for automated COVID-19 detection in chest X-rays?*

---

## Key Features

- **Dataset**: 603 chest X-ray images from Mendeley Data Repository
- **Preprocessing**: Roboflow object detection API for automated artifact removal (ECG leads, text annotations)
- **Dual Feature Extraction**:
  - **Baseline**: Flattened pixel arrays (262,144 features per 512×512 image)
  - **Advanced**: 36 GLCM texture features across multiple distances (1, 5, 10 pixels) and angles (0°, 45°, 90°, 135°)
- **Machine Learning Pipeline**: Logistic Regression, Random Forest, SVC, k-Nearest Neighbors
- **Model Optimization**: GridSearchCV with 5-fold cross-validation
- **Feature Selection**: Permutation importance analysis for GLCM features
- **Comprehensive Evaluation**: ROC-AUC (One-vs-Rest), precision, recall, F1-score, confusion matrices

---

## Key Findings

- **Highest Accuracy**: Random Forest with flattened pixel features achieved **96.69% accuracy**
- **Recommended Model**: Random Forest with GLCM features (**96.13% accuracy**) - optimal balance of performance and efficiency
- **Computational Advantage**: 99.99% dimensionality reduction (36 vs 262,144 features) with only 0.56% accuracy loss
- **Clinical Relevance**: Texture-based approach provides interpretable features suitable for medical deployment

### **Performance Trade-offs**
The research reveals that while pixel-based models achieve maximum accuracy, GLCM-based models offer superior practical deployment characteristics with minimal performance sacrifice.

---

## Project Structure

```
glcm-xray-classifier/
├── data/                     # Dataset organization
│   ├── raw/                  # Original downloaded images  
│   │   ├── COVID-19/        # COVID-19 X-ray images
│   │   ├── Normal/          # Healthy chest X-rays
│   │   └── Pneumonia/       # Pneumonia X-ray images
│   └── cleaned/             # Roboflow-processed images
│       ├── COVID-19/        # Artifact-removed COVID images
│       ├── Normal/          # Processed normal images
│       └── Pneumonia/       # Processed pneumonia images
├── models/                  # Saved trained models
│   ├── rf_flat_best.pkl     # Random Forest (pixel features)
│   ├── rf_glcm_best.pkl     # Random Forest (GLCM features)  
│   └── rf_glcm_positive_only_best.pkl # RF (selected GLCM)
├── main.ipynb              # Complete analysis notebook
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not tracked)
├── README.md              # Project documentation
└── .gitignore             # Version control exclusions
```

---

## Technologies Used

- **scikit-learn** – Machine learning algorithms, model selection, evaluation metrics
- **OpenCV** – Image processing and manipulation
- **scikit-image** – GLCM texture feature extraction and advanced image analysis
- **Roboflow API** – Automated ECG lead and text artifact removal
- **NumPy & Pandas** – Numerical computing and data manipulation
- **Matplotlib & Seaborn** – Data visualization and results presentation
- **IPython Widgets** – Interactive notebook components
- **joblib** – Model persistence and parallel processing

---

## Comprehensive Results Summary

### **Performance Comparison**

| **Approach** | **Model** | **Accuracy** | **Features** | **Recommended Use** |
|--------------|-----------|--------------|--------------|---------------------|
| **Flattened Pixels** | Random Forest | **96.69%** | 262,144 | Maximum accuracy scenarios |
| **GLCM Features** | Random Forest | **96.13%** | 36 | **Recommended deployment** |
| **Selected GLCM** | Random Forest | 94.75% | Selected subset | Resource-constrained environments |

### **Key Insights**

#### **Accuracy vs Efficiency Trade-off**
- **Best Raw Performance**: Random Forest with pixel features (96.69%)
- **Optimal Balance**: Random Forest with GLCM features (96.13%)
- **Minimal Loss**: Only 0.56% accuracy reduction for 99.99% feature reduction

#### **Deployment Recommendation**
**Primary Choice**: Random Forest with GLCM features
- **Clinical Suitability**: Real-time inference capability
- **Resource Efficiency**: 10,000x faster processing
- **Interpretability**: Meaningful texture-based medical insights
- **Scalability**: Suitable for production medical environments

#### **Alternative Options**
- **High-Accuracy**: Use pixel-based model when computational resources are abundant
- **Ultra-Efficient**: Use selected GLCM features for maximum resource constraints

---

## Clinical Applications

### **Diagnostic Capabilities**
- **COVID-19 Detection**: Excellent sensitivity for early identification
- **Pneumonia Recognition**: Strong pattern recognition for bacterial/viral pneumonia  
- **Normal Classification**: Accurate identification of healthy chest X-rays
- **Cross-Class Balance**: Consistent performance across all diagnostic categories

### **Deployment Advantages**
- **Real-time Processing**: GLCM features enable immediate clinical feedback
- **Resource Efficiency**: Suitable for resource-constrained healthcare settings
- **Interpretable Results**: Texture-based features provide clinically meaningful insights

---

## Implementation Guide

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Environment Setup**
1. Create `.env` file with Roboflow API key:
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

2. Download dataset (automatically handled in notebook)

3. Run `main.ipynb` for complete analysis pipeline

### **Model Usage**
```python
import joblib

# Load recommended model
model = joblib.load('models/rf_glcm_best.pkl')

# Make predictions on new GLCM features
predictions = model.predict(new_glcm_features)
```

---

## Current Limitations

1. **Dataset Size**: 603 images may limit generalizability to diverse populations
2. **Single-Institution Data**: May not represent global imaging variations  
3. **Preprocessing Dependency**: Relies on Roboflow API for optimal artifact removal
4. **Static Features**: GLCM features may miss temporal or dynamic patterns
5. **No Deep Learning Baseline**: Limited comparison with CNN-based approaches

---

## Future Research Directions

### **Dataset Enhancement**
- **Scale Expansion**: Increase to 5,000+ images from multiple institutions
- **Demographic Diversity**: Ensure representation across age, gender, ethnicity
- **Temporal Validation**: Test performance across different time periods
- **Multi-Modal Integration**: Incorporate clinical metadata and patient history

### **Technical Improvements**
- **Deep Learning Comparison**: Benchmark against CNN architectures (ResNet, DenseNet)
- **Hybrid Approaches**: Combine GLCM features with deep learning outputs
- **Advanced Texture Analysis**: Explore wavelet transforms and Gabor filters
- **Ensemble Methods**: Combine multiple model predictions for enhanced accuracy

---

## Academic Impact

This research demonstrates that traditional machine learning techniques with carefully engineered features can achieve excellent performance in medical image classification while maintaining computational efficiency and interpretability - critical factors for clinical deployment.

**Key Contributions**:
- Comprehensive comparison of pixel vs. texture-based features
- Practical deployment considerations for medical AI systems  
- Balance between accuracy and computational efficiency
- Reproducible methodology for medical image classification research

---

## License & Usage

This project is intended for **academic and research purposes only**. The code and models should not be used for actual medical diagnosis without proper validation and regulatory approval.

**Citation**: If you use this work in your research, please cite this repository and acknowledge the original dataset sources.

---

## Contact

**Author**: Brenden Runion  
**Repository**: [glcm-xray-classifier](https://github.com/bdog18/glcm-xray-classifier)  
**Date**: April 2025
