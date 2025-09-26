# Brain Tumor MRI Classification Using Deep Learning

## Overview
Brain tumors remain one of the most critical health challenges in neurology. Their early detection and precise classification play a significant role in determining treatment success and patient survival rates. Magnetic Resonance Imaging (MRI) has become the gold standard for brain imaging, providing detailed structural information. However, manual interpretation of MRI scans by radiologists is not only time-consuming but also subject to human error and inter-observer variability.

To address these limitations, this project proposes an automated solution using **deep learning and transfer learning techniques**. Specifically, we employ the **VGG19 architecture**, a powerful convolutional neural network pre-trained on ImageNet, and fine-tune it for the task of brain tumor classification. The model is trained to classify MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

This approach enables efficient, scalable, and highly accurate classification, aiming to support radiologists in clinical decision-making while reducing diagnostic errors.

---

## Dataset
The dataset for this project was created by combining **two publicly available Kaggle datasets**:
1. **Masoudnickparvar Brain Tumor MRI Dataset**  
2. **Indk214 Brain Tumor Dataset**

### Data Preparation
The raw dataset initially contained **over 18,000 MRI images** spanning different classes and formats. To ensure consistency and reliability, several key preprocessing steps were applied:
- **Label unification:** Tumor class labels across the datasets were harmonized (e.g., “pituitary tumor” was mapped to “pituitary”; “notumor” was mapped to “no_tumor”).
- **Duplicate removal:** Duplicate MRI scans were detected using **MD5 hashing** and removed. This step reduced the dataset significantly but ensured data integrity and prevented model overfitting.
- **Balanced splitting:** The dataset was stratified and split into training, validation, and test subsets to maintain class balance across all stages of training.

This preparation ensured a clean, balanced dataset suitable for deep learning experiments while preserving representative diversity across tumor types.

---

## Methodology

### Preprocessing
To make the data compatible with the deep learning model, several preprocessing steps were carried out:
- All images were resized to **224 × 224 pixels**, matching the input requirements of VGG19.
- Images were normalized to values between **0 and 1**, ensuring stable and efficient gradient updates.
- Labels were encoded into integers for model compatibility.

### Data Augmentation
To improve generalization and avoid overfitting, data augmentation techniques were applied:
- Random horizontal flipping  
- Random brightness adjustments  
- Random contrast variations  

These transformations increase the effective size of the dataset and simulate real-world variations in MRI scans.

### Model Architecture
The model was built using **transfer learning with VGG19**:
- **Base Model:** VGG19 pre-trained on ImageNet, with initial layers frozen to retain learned low-level features.
- **Custom Layers:**  
  - Global Average Pooling  
  - Dense layer with 256 neurons (ReLU activation)  
  - Batch Normalization for training stability  
  - Dropout layers for regularization  
  - Final Dense layer with Softmax activation for multi-class classification

This architecture balances computational efficiency with predictive power, allowing the model to adapt to medical imaging while leveraging pre-trained knowledge.

### Training Strategy
- Optimizer: **Adam** with a learning rate of 1e-4 and adaptive scheduling (ReduceLROnPlateau).  
- Loss function: **Sparse Categorical Crossentropy**.  
- Regularization: Early stopping to prevent overfitting and ModelCheckpoint to save the best-performing model.  
- Class imbalance: Handled with **class weighting** computed from the training set.  

---

## Results

The trained model demonstrated outstanding performance on the test dataset.

### Key Metrics:
- **Accuracy:** ~98%  
- **Macro F1-score:** ~0.98  
- **Weighted F1-score:** ~0.98  
- **ROC-AUC:** consistently above 0.97 across all classes  

### Observations:
- The confusion matrix confirmed that misclassifications were extremely rare.  
- ROC curves indicated high discriminative capability between all tumor classes.  
- Glioma, meningioma, and pituitary tumors were classified with high precision, and the model successfully distinguished between “tumor” and “no tumor” cases with strong reliability.  

These results demonstrate that transfer learning with VGG19 is highly effective in brain tumor MRI classification.

---

## Deployment
To showcase the practical usability of this work, the trained model was deployed as a **Streamlit web application**.  

### Workflow:
1. A user uploads an MRI image.  
2. The image undergoes preprocessing (resizing and normalization).  
3. The model predicts the class label.  
4. The result (Glioma, Meningioma, Pituitary, or No Tumor) is displayed clearly on the interface.  

This deployment transforms the research model into an interactive tool that could be used for demonstrations, educational purposes, and future clinical integration.

---

## Significance
This project highlights the role of artificial intelligence in medical imaging by providing:
- **Time efficiency:** MRI scans can be classified in seconds rather than hours.  
- **Diagnostic support:** Reduces radiologist workload and provides a second layer of validation.  
- **Scalability:** The system can be deployed on cloud-based or local applications, making it widely accessible.  

Such a system, when integrated into clinical workflows, has the potential to improve early diagnosis rates and assist in treatment planning.

---

## Future Work
While the project achieved strong results, several areas remain open for further exploration:
- **Tumor segmentation:** Extend the pipeline to not only classify but also segment tumor regions, aiding treatment planning.  
- **Explainability:** Incorporate interpretability techniques such as **Grad-CAM** to provide heatmaps showing which regions influenced the prediction.  
- **External validation:** Test the model on independent clinical datasets to confirm robustness outside of Kaggle data.  
- **Model optimization:** Investigate lightweight architectures suitable for deployment on mobile or edge devices.  

---

## Conclusion
This project successfully developed a deep learning-based classification system for brain tumors using MRI images. The integration of data preprocessing, augmentation, transfer learning, and robust evaluation resulted in a highly accurate and deployable model.  

By achieving ~98% accuracy, the model demonstrates the potential of AI to complement radiologists in diagnostic decision-making. The deployment through Streamlit further emphasizes its practical application, bridging the gap between research and real-world use.  

The results underscore the transformative potential of AI in healthcare, paving the way for future innovations in automated medical diagnostics.
