# 🧠 NeuroCogNet – AI-Powered Dementia Detection & Awareness Platform

An intelligent deep learning-based web application designed to support dementia stage detection, MRI-based condition comparison, and dementia awareness through separate Doctor and Public User modes.

NeuroCogNet combines deep learning predictions, visual explanations, downloadable diagnostic reports, and educational resources in one platform.

---

## 🌟 Project Overview

NeuroCogNet is a deep learning-powered dementia detection platform developed to provide different features based on the selected user mode.

When users open the web application, they are welcomed by a home page where they can choose between:

- 👨‍⚕️ **Doctor Login**
- 🌍 **Public User**

Each mode provides dedicated features to support medical assessment workflows and public awareness.

## 🎯 Key Objectives

- Develop a deep learning-based dementia stage detection system.
- Provide visual explanations through heatmaps.
- Display prediction confidence scores.
- Compare pre-treatment and post-treatment MRI scans.
- Identify whether the condition has improved, worsened, or remained the same based on model outputs.
- Generate downloadable PDF reports.
- Promote dementia awareness through educational content.
- Provide dietary guidance and food portion information for patients.

## 🚀 Key Features

### 1. 👨‍⚕️ Doctor Mode

A dedicated interface with additional assessment features.

**🧠 Dementia Stage Detection**
- Upload MRI images for dementia stage prediction.
- Identify the predicted dementia stage.
- Display prediction confidence scores.
- Generate heatmaps to visually highlight relevant image regions.

**🔬 Pre-Treatment vs Post-Treatment MRI Comparison**
- Upload pre-treatment MRI scans.
- Upload post-treatment MRI scans.
- Compare both MRI inputs.
- Display the comparison outcome:
  - 🟢 Improved
  - 🔴 Worsened
  - 🟡 Same Condition

**📄 Downloadable Diagnosis Report**
- Generate a detailed PDF report.
- Include the available prediction and comparison results.
- Download the report for documentation and reference.

### 2. 🌍 Public User Mode

An educational section designed to improve understanding of dementia and patient care.

**📚 Dementia Awareness**
- Learn about dementia.
- Explore relevant educational information.
- Understand the condition and its impact.

**🥗 Diet & Nutrition Guidance**
- Explore useful food choices for dementia patients.
- Learn about dietary recommendations.
- View suggested daily food portions.

## 🔄 Application Workflow

1. Launch the NeuroCogNet web application.
2. View the home page.
3. Select Doctor Login or Public User.
4. Access the features available in the selected mode.
5. Doctors can perform dementia stage detection, MRI comparison, and PDF report generation.
6. Public users can explore dementia awareness and dietary guidance.

## 🛠️ Tools & Technologies

- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Pre-trained Architecture:** MobileNetV2
- **Web Application:** Python-based web application
- **Image Processing:** MRI image preprocessing
- **Visualization:** Heatmaps and model evaluation visualizations
- **Reporting:** PDF report generation

## 📊 Model Evaluation & Outputs

The project includes model evaluation and visual outputs such as:

- Confusion Matrix
- Model Performance Graphs
- Training and Validation Visualizations
- Dementia Prediction Outputs
- Heatmap Visualizations
- MRI Comparison Results
- Downloadable PDF Report Sample

## 📁 Project Structure

```text
dementia-detection-using-deep-learning/
│
├── Documents/
├── figures/
├── images/
│
├── augment_moderate.py
├── augment_moderate_direct.py
├── evaluate_model.py
├── load_model_test.py
├── train_model.py
├── train_model_pretrained.py
├── testing.py
├── web_app.py
│
├── alzheimer_model_cpu.h5
├── mymobilenetv2_model_finetuned.h5
├── mymobilenetv2_pretrained_finetuned.h5
│
└── README.md
```

## 💡 Skills Demonstrated

- Deep Learning
- Convolutional Neural Networks (CNN)
- Transfer Learning
- MobileNetV2
- Medical Image Classification
- MRI Image Analysis
- Model Evaluation
- Heatmap Visualization
- Python Development
- Web Application Development
- PDF Report Generation

## ⚠️ Disclaimer

NeuroCogNet is an academic project developed for educational and research purposes. Its predictions, heatmaps, and MRI comparisons are experimental outputs and should not be considered medical diagnoses or substitutes for professional clinical evaluation. Dietary information is educational and should not replace individualized medical or nutritional advice.

## 👩‍💻 Developed By

**Sri Rajakavika Palanivel**

M.Sc. Software Systems

GitHub: [SriRajakavika](https://github.com/SriRajakavika)

---

⭐ If you find this project interesting, feel free to explore the repository!
