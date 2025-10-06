# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)


### 📦 1. Dataset Structure & Class Distribution

Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?  
A: Dataset Size: 253 total images
   • Class distribution: 98 no tumor, 155 with tumor
   

Q: What is the class imbalance ratio, and how might this affect model training?  
A: Imbalance ratio: 1.58:1.  This may be considered to be mild imbalance, and in general may not greatly impact training of a CNN model. However the CNN model may be affected due to:
    • Loss Function Bias: Binary cross-entropy gives equal weight to all samples, so the model optimizes mostly for the majority class
    • Gradient Dominance: More samples = more gradient updates favoring that class
    • Decision Boundary Shift: The model's decision boundary shifts toward the majority class
⚠️ Medical context matters: Missing tumors (false negatives) is worse than false alarms. Mitigation/monitoring to include:
    • Data augmentation to increase minority class samples
    • Monitor metrics in addition to accuracy (recall, precision, F1)
    • Use class weights
    • Find optimal decision threshold
    • Look for red flags:
        • ⚠️ Recall for "Tumor" < 0.85 (missing too many tumors)
        • ⚠️ Large gap between precision and recall
        • ⚠️ Val accuracy high but tumor detection poor

### 🖼️ 2. Image Properties & Standardization

Q: What are the different image dimensions present in your dataset?  
A:  Image Size Statistics (n=100 images):
  Width  - Min:  150, Max: 1275, Mean:  352.8
  Height - Min:  168, Max: 1427, Mean:  386.4

Number of unique image sizes: 82

Q: What target image size did you choose for standardization and why?  
A:  128x128  - Faster training, good for initial experiments

Q: What is the pixel intensity range in your raw images?
A:  Pixel Intensity Statistics: 20 images per class analyzed
  WITH Tumor    - Min:   0, Max: 255, Mean:   67.2, Std:   59.0
  WITHOUT Tumor - Min:   0, Max: 255, Mean:   42.2, Std:   49.5
  Significant difference in pixel intensities between classes (p < 0.05)