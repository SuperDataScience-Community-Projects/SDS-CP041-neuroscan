# 🔴 Advanced Track

## ✅ Week 1: Setup + Exploratory Data Analysis (EDA)


### 📦 1. Dataset Structure & Class Distribution

Q: How many images are in the "yes" (tumor) vs "no" (no tumor) classes?  
A: yes: 155, no: 98 

Q: What is the class imbalance ratio, and how might this affect model training?  
A: 98/155 ~> 0.63. This imbalance could lead to a "yes" bias. In other words, the model could become slightly better at identifying 
   images with tumors than it is at identifying images without tumors.


### 🖼️ 2. Image Properties & Standardization

Q: What are the different image dimensions present in your dataset?  
A: The no images width ranges from 150-1920 where the height ranges from 168-1080
   while the "yes" images range from 178-1275 width and 173-1275 height.

Q: What target image size did you choose for standardization and why?  
A: 224x224 to retain as much detail as possible since some images will be downsized by a factor of 5.
   I think this option makes sense as it could reduce noise in the majority while also making the dataset 
   images the same size as the ones used for ResNet. There is the issue of about 20% of the image width's 
   being below the 244 resolution mark which could introduce some noise in the upscale but the alternative
   of downscaling every image to 150x150 would lose a significant amount of data.

Q: What is the pixel intensity range in your raw images?
A: The pixel intensity range is from 0-255



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

