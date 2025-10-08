# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)


### 📦 1. Dataset Structure & Class Distribution

Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?  
A: 
Tumor images:    155
No tumor images:  98
Total images:    253

Q: What is the class imbalance ratio, and how might this affect model training?  
A:
Class Imbalance Ratio (yes:no) = 1.58:1
This doesn't seem that imbalanced. I would expect errors if oversampling by ratios 5:1 or greater

### 🖼️ 2. Image Properties & Standardization

Q: What are the different image dimensions present in your dataset?  
A:  
|        Stat   |      width     |     height    | aspect_ratio |   
|:-------------:|:--------------:|:-------------:|:------------:|
| mean          | 354.237154     | 386.019763    | 0.919033     | 
| std           | 217.111684     | 213.128463    | 0.168495     | 
| min           | 150.000000     | 168.000000    | 0.682454     |
| 25%           | 225.000000     | 248.000000    | 0.815789     | 
| 50%           | 278.000000     | 331.000000    | 0.882500     | 
| 75%           | 400.000000     | 442.000000    | 1.000000     | 
| max           | 1920.000000    | 1427.000000   | 1.785714     | 

Q: What target image size did you choose for standardization and why?  
A:  
I am thinking 244 x 244 as that is a common image size and to pad those that are below that. Downsample the ones above it. 
But now I am wondering if I should try an image size closer the the mean.

Q: What is the pixel intensity range in your raw images?
A:  
Raw Images have a range from 0-255. Need to divide all pixels by 255 to get within range 0-1


## ✅ Week 2–3: CNN Model Development & Training


### 🏗️ 1. CNN Architecture Design

Q: Describe the architecture of your custom CNN model (layers, filters, pooling).  
A:  

Q: Why did you choose this specific architecture for brain tumor classification?  
A:  

Q: How many trainable parameters does your model have?  
A:  


### ⚙️ 2. Loss Function & Optimization

Q: Which loss function did you use and why is it appropriate for this binary classification task?  
A:  

Q: What optimizer did you choose and what learning rate did you start with?  
A:  

Q: How did you configure your model compilation (metrics, optimizer settings)?  
A:  


### 🔄 3. Data Augmentation Strategy

Q: Which data augmentation techniques did you apply and why?  
A:  

Q: Are there any augmentation techniques you specifically avoided for medical images? Why?  
A:  


### 📊 4. Training Process & Monitoring

Q: How many epochs did you train for, and what batch size did you use?  
A:  

Q: What callbacks did you implement (early stopping, learning rate scheduling, etc.)?  
A:  

Q: How did you monitor and prevent overfitting during training?  
A:  


### 🎯 5. Model Evaluation & Metrics

Q: What evaluation metrics did you use and what were your final results?  
A:  

Q: How did you interpret your confusion matrix and what insights did it provide?  
A:  

Q: What was your model's performance on the test set compared to validation set?  
A:  


### 🔄 6. Transfer Learning Comparison (optional)

Q: Which pre-trained model did you use for transfer learning (MobileNetV2, ResNet50, etc.)?  
A:  

Q: Did you freeze the base model layers or allow fine-tuning? Why?  
A:  

Q: How did transfer learning performance compare to your custom CNN?  
A:  


### 🔍 7. Error Analysis & Model Insights

Q: What types of images does your model most commonly misclassify?  
A:  

Q: How did you analyze and visualize your model's mistakes?  
A:  

Q: What improvements would you make based on your error analysis?  
A:

