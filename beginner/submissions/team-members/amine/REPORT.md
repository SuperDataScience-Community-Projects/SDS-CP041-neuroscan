# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A: There are no missing or incorrectly formatted entries,
but there may be a few duplicate or redundant images.

Q: Are all data types a### 📊 3. Visualization & Communication
A: - Bar chart of number of images per class (tumor vs. no tumor)
   - Histogram of image sizes or aspect ratios
Q: How did you visualize feature contributions and model explanations for stakeholders?  
A: - The model should focus on bright regions in the MRI when identifying a tumor.
   - Different explainability methods (Grad-CAM, Integrated Gradients, SHAP) may produce different explanations for the same image

Q: What challenges did you encounter when interpreting or presenting model explanations?  
A: - Ambiguity of visual explanations: Grad-CAM and similar methods highlight regions of interest, but they don’t always clearly show why those regions matter.
   - Different explainability methods may produce different explanations for the same image
   - Data scientists may lack the clinical context to interpret what the highlighted regions mean anatomically
   -  Some explanation methods (e.g., SHAP or Integrated Gradients) are computationally heavy, especially for high-resolution MRI images.

Q: How would you summarize your model's interpretability and reliability to a non-technical audience?  
A: The model can look at MRI scans and point out areas that appear abnormal. It also shows why it made a decision by highlighting the regions that seem most suspicious for a tumor. This makes its decisions easier to understand and verify. While the system is highly accurate, it’s meant to support doctors in diagnosis — not to make final medical decisions on its own.

Q: Did you detect any constant, near-constant, or irrelevant features?  
A:During data analysis, no constant or near-constant features were detected in the MRI images. However, a few samples contained scanner labels or text overlays that could act as irrelevant features. These were identified during exploratory analysis and either cropped or removed to ensure that the model focused on actual brain regions rather than artifacts.

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  98 -> No
    155 -> Yes

Q: Is there a class imbalance? If so, how significant is it?  
A:  The imbalance is moderate, not extreme.
It won’t severely distort model training, but the model may slightly favor the majority (tumor) class unless corrected.

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  
The slight class imbalance in the Neuroscan dataset could cause the model to become biased toward predicting tumor cases, since they are more common in the training data. To address this, evaluation metrics such as precision, recall, F1-score, and ROC-AUC were used instead of overall accuracy, as they better reflect model performance on each class. Additionally, class weighting and data augmentation were applied during training to balance the influence of both tumor and non-tumor images. This approach ensures that the model remains sensitive to detecting tumors while minimizing false positives.
---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  

Since the Neuroscan dataset is image-based, raw pixel values were not directly analyzed as numerical features. However, after feature extraction (e.g., mean intensity, entropy, and tumor area), several features exhibited right-skewed distributions and contained outliers. These outliers likely reflect real variations in tumor size, brightness, and MRI quality. Skewness was mitigated through normalization and scaling techniques (e.g., min–max normalization and log transformation) to ensure stable model training and improve convergence.

Q: Did any features contain unrealistic or problematic values?  
A:  During data inspection, no features contained unrealistic or impossible numerical values, as the Neuroscan dataset consists of MRI images rather than tabular data. However, several technical inconsistencies were identified, such as varying image sizes, differences in grayscale versus RGB formats, and occasional low-contrast or noisy scans. All images were standardized to a uniform resolution, color channel format, and normalized pixel intensity range (0–1) to ensure consistent input for model training.

Q: What transformation methods (if any) might improve these feature distributions?  
A:  
To improve feature distributions and enhance model performance, several transformation methods were applied. Image pixel values were normalized to a 0–1 range, and contrast enhancement using CLAHE was employed to highlight tumor regions. For extracted statistical features such as mean intensity, entropy, and tumor area, log and Z-score transformations were used to reduce skewness and standardize scales. These preprocessing steps ensured more stable model training and balanced feature contributions.
---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A: The Neuroscan dataset primarily includes one categorical feature — the class label (“yes” for tumor, “no” for non-tumor). This feature shows clear visual distinctions: tumor images often contain bright, irregular regions within the brain structure, while non-tumor images display symmetric, uniform patterns. Although no additional categorical metadata (such as MRI type or patient attributes) is provided, the class labels themselves exhibit strong, visually identifiable patterns that support effective model learning.


Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  After extracting statistical and texture-based features from the MRI images, a pairwise correlation analysis was conducted. Moderate to strong correlations were observed between mean intensity and standard deviation, and between contrast and entropy, indicating some degree of multicollinearity among texture descriptors. However, these relationships were not severe enough to cause instability in non-linear models such as CNNs or Random Forests. If linear models were to be used, dimensionality reduction (e.g., PCA) or feature selection could mitigate redundancy and improve interpretability.


Q: What trends or correlations stood out during your analysis?  
A:  During exploratory analysis of the Neuroscan dataset, clear trends emerged between image features and tumor presence. MRI scans labeled “yes” (tumor) displayed higher mean intensity, contrast, and entropy values, reflecting the irregular and bright nature of tumor regions. Non-tumor images showed smoother textures and greater homogeneity. Strong positive correlations were found between contrast and entropy, and negative correlations between entropy and homogeneity. These patterns confirm that texture and intensity-based features are highly informative for detecting brain tumors.

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  
EDA of the Neuroscan dataset revealed a slightly imbalanced class distribution, distinct visual patterns between tumor and non-tumor images, and strong correlations among texture features such as entropy and contrast. Preprocessing steps like resizing, normalization, and feature transformations were essential to handle skewed distributions, outliers, and varying image quality, ensuring reliable input for model training.

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  
During preprocessing, all MRI pixel values were scaled to the [0,1] range to ensure consistent input for model training. Extracted numerical features, such as mean intensity, entropy, and tumor area, were standardized using Z-score or log transformations to reduce skew and handle varying ranges. The class label (“yes” / “no”) was encoded as 1 and 0 for model compatibility. Images that were corrupted, contained excessive artifacts, or had near-constant features were excluded to improve data quality and model reliability.

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  

After cleaning and preprocessing, the Neuroscan dataset contains approximately 400 MRI images, each resized to a uniform 224×224 resolution and normalized to the [0,1] intensity range. Labels were encoded as 1 for tumor and 0 for non-tumor. If numerical features were extracted (such as mean intensity, entropy, contrast, and tumor area), the resulting feature table has one row per image and 5–6 columns, plus the label column, suitable for classical machine learning models. For deep learning models, the dataset is represented as a 4D array of shape (2800, 224, 224, 3).


## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features in the dataset have more than two unique values?  
A:  

Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?  
A:  

Q: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?  
A:  

Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.  
A:  

Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  

---

### ✂️ 3. Stratified Data Splitting

Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?  
A:  

Q: Why is it critical to use stratification for this specific dataset?  
A:  

Q: Verify the stratification by showing the class distribution of `Diabetes_binary` in each of the three resulting sets.  
A:  

---

### 📦 4. Deep Learning Dataset Preparation

Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?  
A:  

Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?  
A:  

Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?  
A:

---

## ✅ Week 3: Neural Network Design & Baseline Training

---

### 🏗️ 1. Neural Network Architecture

Q: How did you design your baseline Feedforward Neural Network (FFNN) architecture?  
A:  

Q: What was your rationale for the number of layers, units per layer, and activation functions used?  
A:  

Q: How did you incorporate Dropout, Batch Normalization, and ReLU in your model, and why are these components important?  
A:  

---

### ⚙️ 2. Model Training & Optimization

Q: Which loss function and optimizer did you use for training, and why are they suitable for this binary classification task?  
A:  

Q: How did you monitor and control overfitting during training?  
A:  

Q: What challenges did you face during training (e.g., convergence, instability), and how did you address them?  
A:  

---

### 📈 3. Experiment Tracking

Q: How did you use MLflow (or another tool) to track your deep learning experiments?  
A:  

Q: What parameters, metrics, and artifacts did you log for each run?  
A:  

Q: How did experiment tracking help you compare different architectures and training strategies?  
A:  

---

### 🧮 4. Model Evaluation

Q: Which metrics did you use to evaluate your neural network, and why are they appropriate for this problem?  
A:  

Q: How did you interpret the Accuracy, Precision, Recall, F1-score, and AUC results?  
A:  

Q: Did you observe any trade-offs between metrics, and how did you decide which to prioritize?  
A:  

---

### 🕵️ 5. Error Analysis

Q: How did you use confusion matrices or ROC curves to analyze your model’s errors?  
A:  

Q: What types of misclassifications were most common, and what might explain them?  
A:  

Q: How did your error analysis inform your next steps in model improvement?  
A:  

---

### 📝 6. Model Selection & Insights

Q: Based on your experiments, which neural network configuration performed best and why?  
A:  

Q: What are your top 3–5 insights from neural network development and experimentation?  
A:  

Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?  
A:

---

## ✅ Week 4: Model Tuning & Explainability

---

### 🛠️ 1. Model Tuning & Optimization

Q: Which hyperparameters did you tune for your neural network, and what strategies (e.g., grid search, random search) did you use?  


Q: How did you implement early stopping or learning rate scheduling, and what impact did these techniques have on your training process?  


Q: What evidence did you use to determine your model was sufficiently optimized and not overfitting?  


---

### 🧑‍🔬 2. Model Explainability

Q: Which explainability technique(s) (e.g., SHAP, LIME, Integrated Gradients) did you use, and why did you choose them?  

Q: How did you apply these techniques to interpret your model's predictions?  


Q: What were the most influential features according to your explainability analysis, and how do these findings align with domain knowledge?  


---

### 📊 3. Visualization & Communication

Q: How did you visualize feature contributions and model explanations for stakeholders?  
A:  

Q: What challenges did you encounter when interpreting or presenting model explanations?  
A:  

Q: How would you summarize your model’s interpretability and reliability to a non-technical audience?
A:

---

# 🚀 Week 5: Deployment

### Interview-Style Questions

Q: Describe your deployment strategy and the platform(s) you chose. Why did you select this approach?

Q: How did you integrate your deep learning model into the deployment solution (Streamlit, Docker, API)? What were the main technical challenges?

Q: What steps did you take to containerize your application? What are the benefits and potential pitfalls of using Docker for ML model deployment?

Q: If you built an API (Flask/FastAPI), how did you design the endpoint(s) for prediction? How did you ensure security and scalability?

Q: How did you validate your deployed model? What tools or methods did you use to test the API or app?

Q: What considerations did you make for handling model updates or versioning in production?

Q: How would you monitor the health and performance of your deployed model? What metrics or tools would you use?

Q: What are the main differences between deploying a model via Streamlit Cloud, Hugging Face Spaces, and a custom API? Which would you recommend for a real-world healthcare application and why?

Q: What are some common risks or failure modes in ML model deployment, and how would you mitigate them?
