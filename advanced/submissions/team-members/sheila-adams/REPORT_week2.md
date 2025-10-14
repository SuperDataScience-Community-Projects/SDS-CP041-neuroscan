# 🔴 Advanced Track

## ✅ Week 2–3: CNN Model Development & Training

### Issues encountered and solutions
Found duplicate images distributed between training and validation sets
Created content-based hashes to check for and prevent overlaps and data leakage

### 🏗️ 1. CNN Architecture Design

Q: Describe the architecture of your custom CNN model (layers, filters, pooling).  
A:  I ran models with two architectures: (inputs in both were 128 x 128)
I started with three convolutional blocks with max pooling, 2 dense layers, 1 final output layer, and Adam optimizer with a learning rate of 0.001. It also has a global pooling step prior to the dense layer addition. 

The first model started with a filter base of 32 and the first convolutional block had 2 filter layers.  The second, simpler model, started with a filter base of 16 and dropped the 2nd layer of filters in the 1st convolutional block. The dropout rate was 25% in the first baseline model, but was decreased to 15% for the simpler model.

Q: Why did you choose this specific architecture for brain tumor classification?  
A: This was recommended based on the convolutional filters' ability to accept grayscale images, determine horizontal edges of the images from dark to bright, and learn patterns from the image data, moving from simple to more complex patterns. The max pooling reduces the image size, maintaining the key information to keep the strongest features while filtering out image noise. The dropout rate was based on the level of turning off or masking some of the neurons to avoid memorization and overfitting. 

Q: How many trainable parameters does your model have?  
A:  Total params: 355,809 (1.36 MB)
    Trainable params: 354,145 (1.35 MB)
    Non-trainable params: 1,664 (6.50 KB)


### ⚙️ 2. Loss Function & Optimization

Q: Which loss function did you use and why is it appropriate for this binary classification task?  
A: Binary cross-entropy, because it is designed for binary classification (sigmoid final output), creates smooth, effective gradients, penalizes mistakes appropriately.

It has an intuitive interpretation:
- Good prediction → low surprise → low loss
- Bad prediction → high surprise → high loss
- Confident bad prediction → VERY high surprise → VERY high loss

Q: What optimizer did you choose and what learning rate did you start with?  
A: Adam for its approach of giving each parameter its own learning rate and treating early layers with smaller adjustments, and later decision layers with larger adjustments. I started with a learning rate of 0.001 and never adjusted this as it was considered optimal for both of the models I tested.

Q: How did you configure your model compilation (metrics, optimizer settings)?  
A:  I configured it to output accuracy, precision, recall, and AUC.


### 🔄 3. Data Augmentation Strategy

Q: Which data augmentation techniques did you apply and why?  
A:  keras.layers augmentation, claimed to be the modern TensorFlow standard, with no generators needed. 

Q: Are there any augmentation techniques you specifically avoided for medical images? Why?  
A: I did not deeply research other techniques - only those suited for medical images.


### 📊 4. Training Process & Monitoring

Q: How many epochs did you train for, and what batch size did you use?  
A: I have used 50 epochs so far, but built in patience between 7-10 for early stopping. A have used a batch size of 32.

Q: What callbacks did you implement (early stopping, learning rate scheduling, etc.)?  
A: I integrated early stopping, and in the hyperparameter tuning step I integrated variation in dropout rates and a learning rate scheduler, none of which improved performance.

Q: How did you monitor and prevent overfitting during training?  
A: It has been evident from looking at a range of metrics that the model is overfitting during the training.  My best assessment is that the data set is too small.


### 🎯 5. Model Evaluation & Metrics

Q: What evaluation metrics did you use and what were your final results?  
A: During one of the runs I got recall of >95%, but specificity of 72%, precision < 80% and AUC of 83%. All of the other trials have shown a tumor recall of 1.0 which indicates a model issue. I am trying to troubleshoot if this could be other than too few samples in the training set.

Q: How did you interpret your confusion matrix and what insights did it provide?  
A: It was clear from the confusion matrix that the model is no better than random guessing.  It is not learning from the data.

Q: What was your model's performance on the test set compared to validation set?  
A:  NA


### 🔄 6. Transfer Learning Comparison (optional)

Q: Which pre-trained model did you use for transfer learning (MobileNetV2, ResNet50, etc.)?  
A: I am going to start with MobileNetV2

Q: Did you freeze the base model layers or allow fine-tuning? Why?  
A: TBD, week 3

Q: How did transfer learning performance compare to your custom CNN?  
A: TBD, week 3


### 🔍 7. Error Analysis & Model Insights

Q: What types of images does your model most commonly misclassify?  
A:  TBD, I started this in another notebook where I explored augmentation a bit.

Q: How did you analyze and visualize your model's mistakes?  
A:  confusion matrix, training metrics plots, summary print statements

Q: What improvements would you make based on your error analysis?  
A: I want to learn a bit more about the Conv2D layers and the effect of dropout rate. I compared my output with those who posted analyses on the Kaggle site for this data set, and none had good performance with CNN. I would like to try transfer learning next.

