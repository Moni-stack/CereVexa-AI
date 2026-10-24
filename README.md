# 🧠 Cerevexa AI
### AI-Powered Brain Tumor Detection, Localization & Explainable MRI Analysis

Cerevexa AI is a deep learning-based medical imaging project designed to classify brain tumors from MRI scans, highlight suspicious regions using Grad-CAM, and visualize tumor regions through image segmentation.

The project combines **Convolutional Neural Networks (CNNs), U-Net segmentation, and Explainable AI (XAI)** to make brain MRI analysis more interpretable through visual insights.

---

<img width="2560" height="1600" alt="Screenshot (1698)" src="https://github.com/user-attachments/assets/084baab8-a735-4b01-8d23-282d9cf344a2" />


## ✨ Key Features

- **Brain Tumor Classification** — Classifies MRI scans into three categories: Glioma, Meningioma, and Pituitary Tumor.
- **Tumor Segmentation** — Uses a U-Net-based model to identify and outline suspected tumor regions.
- **Grad-CAM Visualization** — Highlights image regions that contribute to the classification model's prediction.
- **Prediction Confidence** — Displays the model's predicted class and associated confidence score.
- **Interactive Web Application** — Provides an intuitive interface built with Streamlit for uploading MRI images and viewing results.
- **Explainable AI** — Combines classification results with visual explanations to help users better understand model predictions.

> **Note:** These features depend on the successful integration of the trained classification and segmentation models into the application.

---

## 🎯 Project Objectives

1. Develop a deep learning model to classify brain MRI scans into tumor categories.
2. Implement image segmentation to identify and visualize suspected tumor regions.
3. Integrate Grad-CAM to provide visual explanations of classification predictions.
4. Build an interactive web application for presenting model outputs.
5. Explore how explainable AI can improve the interpretability of medical image analysis systems.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning model development and inference |
| ResNet18 | Brain tumor classification |
| U-Net | Tumor region segmentation |
| Grad-CAM | Visualization of classification-related image regions |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| Matplotlib | Visualization of MRI images and model outputs |
| Google Colab | Model training and experimentation |
| Visual Studio Code | Application development and model integration |

---

## 🧠 Models and Methodology

### 1. Brain Tumor Classification — ResNet18

A ResNet18-based convolutional neural network is trained to classify MRI images into three categories:

- Glioma
- Meningioma
- Pituitary Tumor

The model learns visual patterns from labeled MRI images and produces class scores that can be converted into probabilities for presentation.

### 2. Tumor Segmentation — U-Net

A U-Net-based segmentation model is used to predict pixel-level tumor regions.

Unlike classification, which predicts a category for an entire image, segmentation identifies the pixels that the model considers part of the tumor region.

The predicted mask can be overlaid on the original MRI image to visualize the suspected region.

### 3. Explainable AI — Grad-CAM

Grad-CAM generates a heatmap highlighting image regions that influence the classification model's prediction.

This visualization helps users examine which parts of the MRI image contributed to the model's output.

**Important distinction:** Grad-CAM provides a model-explanation heatmap, whereas U-Net produces a segmentation mask. They serve different purposes and should not be treated as interchangeable.

---

## 📂 Project Structure

The following is a suggested project structure. Adjust the filenames to match your implementation.

```text
Cerevexa-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── resnet18_best.pth
│   ├── unet_model.safetensors
│   └── config.json
│
├── src/
│   ├── classifier.py
│   ├── segmentation.py
│   ├── gradcam.py
│   └── preprocessing.py
│
├── utils/
│   └── visualization.py
│
├── assets/
│   └── screenshots/
│
└── notebooks/
    ├── classification_training.ipynb
    └── segmentation_training.ipynb
```

**Note:** The filenames above are examples. The application must use the correct checkpoint format, architecture, and preprocessing configuration for each trained model.

---

## 🚀 Getting Started

### Prerequisites

- Python installed on your system
- Visual Studio Code or another Python IDE
- Git (optional)
- Trained model checkpoints
- The dependencies required by your implementation

Model training can be performed in Google Colab, while the Streamlit application can be developed and run locally.

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Cerevexa-AI
```

Replace the placeholder with your repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that your `requirements.txt` contains all the libraries used by your actual application, including compatible PyTorch and torchvision versions.

### 4. Add Trained Model Checkpoints

Place the trained classification and segmentation checkpoints in the appropriate locations under `models/`.

Verify that your application loads the models using their correct architectures and preprocessing settings.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

For additional setup guidance, refer to the official [Streamlit documentation](https://docs.streamlit.io/get-started) and [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🔄 Application Workflow

1. **Upload MRI:** The user uploads a brain MRI image.
2. **Preprocessing:** The image is converted into the format expected by the trained models.
3. **Classification:** ResNet18 predicts the tumor category.
4. **Confidence Display:** The application displays the predicted category and associated model confidence.
5. **Grad-CAM:** A heatmap visualizes regions influencing the classification prediction.
6. **Segmentation:** U-Net predicts a mask representing the suspected tumor region.
7. **Visualization:** The application presents the original MRI alongside the available heatmap and segmentation results.

---

## 📊 Dataset

The project uses the **Figshare Brain Tumor Dataset**, which contains brain MRI images, tumor category labels, and segmentation masks.

The dataset includes three tumor categories:

- Glioma
- Meningioma
- Pituitary Tumor

The images and their corresponding masks can support classification and segmentation experiments.

Dataset preprocessing, train-validation-test splitting, and model evaluation should be performed carefully to reduce data leakage and obtain reliable results.

### Dataset Preparation

- Extract MRI images and associated labels.
- Load the corresponding tumor masks.
- Resize and normalize images according to model requirements.
- Prepare classification labels and segmentation targets.
- Split the data into training, validation, and test sets.
- Keep the test set separate from model training and hyperparameter selection.

**Dataset attribution:** Please include the original dataset source, citation, and applicable license information in your repository.

---

## 📈 Evaluation Metrics

The models can be evaluated using the following metrics.

| Task | Evaluation Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Segmentation | Dice Score, Intersection over Union (IoU), Precision, Recall |
| Explainability | Qualitative inspection of Grad-CAM heatmaps |

Classification accuracy alone does not establish clinical reliability. Segmentation quality and the behavior of the models on unseen data should also be assessed.

---

## 🔮 Future Enhancements

- Improve classification performance through data augmentation and hyperparameter tuning.
- Experiment with alternative CNN architectures.
- Refine segmentation performance and tumor boundary visualization.
- Add side-by-side MRI, Grad-CAM, and segmentation comparisons.
- Introduce model evaluation reports and prediction history.
- Improve interface accessibility and usability.
- Investigate additional explainability techniques.
- Evaluate model robustness on independent datasets.

---

## ⚠️ Disclaimer

Cerevexa AI is an educational and research-oriented project developed to explore deep learning and explainable AI in medical image analysis.

It is **not a clinically validated diagnostic system** and must not be used as a substitute for professional medical evaluation, radiological interpretation, or clinical decision-making.

Predictions, confidence scores, heatmaps, and segmentation masks may be incorrect. They should not be used independently to diagnose a medical condition or determine treatment.

---

## 👩‍💻 Project Information

**Project Name:** Cerevexa AI  
**Domain:** Artificial Intelligence · Deep Learning · Medical Image Analysis  
**Primary Models:** ResNet18 and U-Net  
**Explainability Technique:** Grad-CAM  
**Interface:** Streamlit  
**Development Environment:** Google Colab and Visual Studio Code

---

## ⭐ Acknowledgements

This project builds on open-source deep learning tools and publicly available brain MRI research datasets.

Thanks to the researchers and developers who make medical imaging datasets, machine learning frameworks, and explainability techniques available for educational and research purposes.

---

*Built with Python, Deep Learning, and Explainable AI.*
