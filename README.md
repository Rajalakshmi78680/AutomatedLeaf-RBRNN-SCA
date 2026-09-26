# AutomatedLeaf-RBRNN-SCA
This project presents an automated plant leaf disease classification framework
# An Automated Plant Leaf Disease Classification Framework Using Three-Step Feature Extraction-Based Residual Bi-RNN with Spatial Channel Attention

![Plant Disease](https://img.shields.io/badge/Application-Plant%20Disease%20Detection-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%2B%20Bi--RNN-blue)
![Attention](https://img.shields.io/badge/Attention-Spatial--Channel-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Leaf%20Analysis-purple)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Agriculture AI](https://img.shields.io/badge/Agriculture-AI-success)

## 📌 Overview

This project presents an **automated plant leaf disease classification framework** that combines optimized image segmentation, multi-level feature extraction, feature fusion, residual bidirectional recurrent learning, and spatial-channel attention.

The proposed framework addresses the limitations of conventional plant disease identification methods such as manual inspection, laboratory analysis, and conventional image-classification approaches.

The complete framework consists of five major stages:

```text
Plant Leaf Images
       │
       ▼
Image Pre-processing
       │
       ▼
SA-POA Optimized Leaf Segmentation
       │
       ▼
Three-Step Feature Extraction
       │
       ├── Step 1: CNN Deep Features
       ├── Step 2: LBP + LGP Texture Features
       └── Step 3: GLCM + Shape Features
       │
       ▼
Feature Fusion
       │
       ▼
Residual Bi-RNN
       │
       ▼
Spatial Channel Attention
       │
       ▼
Plant Disease Classification
```

The published study reports approximately **95% accuracy**, with the proposed RBRNN-SCA architecture designed to capture complementary deep, texture, statistical, and shape information from leaf images.

---

# 🎯 Research Objectives

The primary objectives of this project are:

1. Develop an automated plant leaf disease classification system.
2. Reduce dependence on manual visual inspection.
3. Improve leaf-region segmentation using adaptive optimization.
4. Extract complementary deep and handcrafted image features.
5. Combine CNN, LBP, LGP, GLCM, and shape descriptors.
6. Develop a Residual Bi-directional RNN classification architecture.
7. Integrate Spatial Channel Attention for discriminative feature selection.
8. Reduce redundant information through feature fusion.
9. Improve classification accuracy and robustness.
10. Support intelligent agriculture and smart-farming applications.

---

# 🌱 Problem Statement

Plant diseases can significantly affect crop quality and agricultural productivity. Disease symptoms may appear as spots, discoloration, texture changes, lesions, or morphological abnormalities on leaves.

Traditional approaches can be:

* Time-consuming
* Subjective
* Dependent on expert knowledge
* Difficult to scale
* Unsuitable for continuous large-scale monitoring

Computer vision provides an automated alternative:

```text
             Plant Leaf
                 │
                 ▼
          Image Acquisition
                 │
                 ▼
        Disease Symptoms
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     Color     Texture    Shape
       │         │         │
       └─────────┼─────────┘
                 ▼
          Disease Pattern
                 │
                 ▼
        AI Classification
```

The proposed framework combines multiple feature representations to improve recognition of plant disease patterns.

---

# 🏗️ Proposed Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                  PLANT LEAF IMAGE                          │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │ Image Pre-processing │
                 │ Contrast Enhancement │
                 │ Color Transformation │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Leaf Segmentation    │
                 │ Optimal Thresholding  │
                 │ + SA-POA             │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Three-Step Feature         │
              │ Extraction                 │
              └─────────────┬──────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
    ┌──────────┐      ┌────────────┐    ┌─────────────┐
    │   CNN    │      │ LBP + LGP  │    │ GLCM + Shape│
    │ Features │      │  Features  │    │  Features   │
    └────┬─────┘      └─────┬──────┘    └──────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ Feature Fusion   │
                  └────────┬─────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Residual Bi-RNN   │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Spatial Channel    │
                │ Attention (SCA)    │
                └──────────┬─────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Fully Connected │
                  │ Classification   │
                  └────────┬────────┘
                           │
                           ▼
                    Disease Class
```

The published architecture consists of data collection, preprocessing, leaf segmentation, three-step feature extraction, and classification using RBRNN-SCA.

---

# 🖼️ Stage 1 — Image Acquisition

Plant leaf images are collected from benchmark datasets.

The published study uses a dataset containing **44 plant-leaf classes** for classification experiments.

The image acquisition pipeline is:

```text
Dataset
   │
   ▼
Plant Leaf Images
   │
   ├── Healthy Leaves
   └── Diseased Leaves
            │
            ▼
      Image Processing
```

The framework can also be adapted to other plant disease datasets such as:

* PlantVillage
* Rice Leaf Disease datasets
* Tomato disease datasets
* Potato disease datasets
* Apple leaf disease datasets
* Custom field-acquired datasets

---

# 🧹 Stage 2 — Image Pre-processing

Raw images may contain:

* Background noise
* Illumination variations
* Low contrast
* Color variations
* Image artifacts

The preprocessing stage improves the quality of the input image.

```text
Raw Image
    │
    ▼
Noise / Quality Processing
    │
    ▼
Contrast Enhancement
    │
    ▼
Color Model Transformation
    │
    ▼
Pre-processed Leaf Image
```

The published methodology uses contrast enhancement and color-model transformation during preprocessing.

---

# ✂️ Stage 3 — Leaf Segmentation

The objective of segmentation is to separate the relevant leaf region from the background.

```text
Original Image
      │
      ▼
Threshold Estimation
      │
      ▼
Optimal Binary Threshold
      │
      ▼
Leaf Region
      │
      ▼
Disease-Relevant Region
```

The threshold parameter is optimized using the **Self-Adaptive Pelican Optimization Algorithm (SA-POA)**.

---

# 🦩 Self-Adaptive Pelican Optimization Algorithm

SA-POA is used to optimize the segmentation threshold.

Conceptually:

```text
                Candidate Thresholds
                         │
                         ▼
                 ┌──────────────┐
                 │    SA-POA    │
                 └──────┬───────┘
                        │
                        ▼
                 Evaluate Fitness
                        │
                        ▼
                 Update Search
                        │
                        ▼
                 Optimal Threshold
                        │
                        ▼
                Binary Segmentation
```

The optimized threshold is intended to provide cleaner leaf regions and improve subsequent feature extraction.

---

# 🔬 Three-Step Feature Extraction

One of the major contributions of the framework is the combination of three different feature groups.

```text
                 Segmented Leaf
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
       Step 1        Step 2        Step 3
          │            │             │
          ▼            ▼             ▼
        CNN        LBP + LGP     GLCM + Shape
          │            │             │
          └────────────┼─────────────┘
                       ▼
                 Feature Fusion
```

The three feature sets are designed to provide complementary information about the disease patterns.

---

# 🧠 Step 1 — CNN Deep Features

Convolutional Neural Networks are used to learn high-level visual representations.

CNN features can capture:

* Disease patterns
* Lesion structures
* Color variations
* Local spatial patterns
* Shape-related visual information
* High-level semantic characteristics

Conceptually:

```text
Leaf Image
    │
    ▼
Convolution
    │
    ▼
Feature Maps
    │
    ▼
Pooling
    │
    ▼
Deep Feature Representation
```

Let the CNN representation be:

$$
F_{CNN}=CNN(X)
$$

where \(X\) is the segmented leaf image.

---

# 🧩 Step 2 — LBP and LGP Features

The second feature group uses handcrafted texture descriptors.

## Local Binary Pattern

LBP captures local texture patterns by comparing a central pixel with its neighborhood.

A simplified representation is:

$$
LBP=
\sum_{p=0}^{P-1}
s(g_p-g_c)2^p
$$

where:

* \(g_c\) = central pixel
* \(g_p\) = neighboring pixel
* \(P\) = number of neighbors

LBP can identify:

* Spots
* Roughness
* Fine textures
* Local intensity variations

---

## Local Gradient Pattern

LGP captures local gradient characteristics of the image.

It provides additional information about:

* Edge patterns
* Gradient transitions
* Texture variations
* Disease boundaries

The combined representation is:

$$
F_{texture}=[F_{LBP},F_{LGP}]
$$

---

# 📊 Step 3 — GLCM and Shape Features

The third feature group contains statistical and morphological information.

## Gray-Level Co-occurrence Matrix

GLCM describes spatial relationships between pixel intensities.

Typical GLCM features include:

* Contrast
* Correlation
* Energy
* Homogeneity
* Entropy

For example:

$$
Contrast =
\sum_{i,j}(i-j)^2P(i,j)
$$

GLCM is useful for identifying disease-related texture changes.

---

## Shape Features

Shape descriptors capture morphological characteristics such as:

* Area
* Perimeter
* Circularity
* Aspect ratio
* Compactness
* Boundary characteristics

The third feature group can be represented as:

$$
F_{shape}=[F_{GLCM},F_{Shape}]
$$

---

# 🔗 Feature Fusion

The three feature groups are combined into a unified feature representation.

$$
F_s =
\{
F_{CNN},
F_{LBP/LGP},
F_{GLCM/Shape}
\}
$$

Conceptually:

```text
CNN Features
     │
     ├───────────────┐
     │               │
LBP + LGP            │
     │               │
     ├───────────────┤
     │               │
GLCM + Shape         │
     │               │
     └───────┬───────┘
             ▼
       Feature Fusion
             │
             ▼
      Unified Feature Vector
```

The purpose is to combine:

```text
Deep Features
      +
Texture Features
      +
Statistical Features
      +
Shape Features
      =
Rich Disease Representation
```

---

# 🔄 Residual Bi-directional RNN

The fused features are supplied to a **Residual Bi-directional Recurrent Neural Network (RBRNN)**.

A conventional Bi-RNN processes information in two directions:

```text
Forward:

x1 → x2 → x3 → x4 → x5


Backward:

x5 → x4 → x3 → x2 → x1
```

The forward and backward representations are combined:

$$
h_t=[\overrightarrow{h_t};
\overleftarrow{h_t}]
$$

This allows the model to use contextual information from both directions.

---

# ➕ Residual Connections

Residual connections provide direct paths between network layers.

```text
                 ┌──────────────────────┐
                 │                      │
Input ───────────┼────────► Bi-RNN ─────┼──► Output
                 │             │        │
                 │             │        │
                 └─────────────┘        │
                    Residual Path       │
```

A simplified residual formulation is:

$$
H_{out}=H_{BiRNN}+H_{input}
$$

Residual learning can help maintain information flow through deeper recurrent structures and reduce difficulties associated with gradient propagation.

---

# 🎯 Spatial Channel Attention

The RBRNN is enhanced using **Spatial Channel Attention (SCA)**.

The purpose of SCA is to emphasize informative features and suppress less relevant activations.

```text
              Feature Tensor
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    Spatial Attention   Channel Attention
          │                   │
          └─────────┬─────────┘
                    ▼
             Attention Maps
                    │
                    ▼
           Feature Re-weighting
                    │
                    ▼
           Enhanced Features
```

The published SCA design uses cross-dimensional interactions between spatial and channel dimensions and generates attention maps that are applied to the input representation.

---

# 🧠 RBRNN-SCA Classifier

The final classifier integrates:

```text
Feature Fusion
      │
      ▼
Residual Bi-RNN
      │
      ▼
Spatial Channel Attention
      │
      ▼
Attention-weighted Representation
      │
      ▼
Fully Connected Layer
      │
      ▼
Softmax
      │
      ▼
Disease Class
```

The final prediction can be represented as:

$$
\hat{y}=Softmax(WF_{SCA}+b)
$$

---

# 🧮 Hidden Neuron Optimization

The hidden-neuron configuration of the RBRNN is optimized using SA-POA.

```text
Candidate Hidden Neurons
          │
          ▼
       SA-POA
          │
          ▼
     Model Training
          │
          ▼
    Validation Fitness
          │
          ▼
 Optimal Hidden-Neuron Count
```

This provides an optimization layer for the recurrent classification architecture.

---

# 🔄 Complete Algorithm

```text
INPUT:
    Plant Leaf Image

OUTPUT:
    Predicted Disease Class

1. Acquire plant leaf image
2. Apply image preprocessing
3. Enhance image contrast
4. Transform color representation
5. Generate candidate segmentation thresholds
6. Optimize threshold using SA-POA
7. Perform optimal binary segmentation
8. Extract CNN deep features
9. Extract LBP features
10. Extract LGP features
11. Extract GLCM features
12. Extract shape features
13. Normalize feature groups
14. Fuse the three feature sets
15. Initialize Bi-RNN
16. Add residual connections
17. Apply Spatial Channel Attention
18. Optimize hidden-neuron configuration
19. Pass enhanced features to fully connected layer
20. Apply Softmax classification
21. Generate predicted plant disease
```

---

# 📊 Performance Evaluation

The model can be evaluated using several classification metrics.

## Accuracy

$$
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
$$

## Precision

$$
Precision=
\frac{TP}
{TP+FP}
$$

## Sensitivity / Recall

$$
Sensitivity=
\frac{TP}
{TP+FN}
$$

## Specificity

$$
Specificity=
\frac{TN}
{TN+FP}
$$

## F1-Score

$$
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
$$

## Matthews Correlation Coefficient

$$
MCC=
\frac{
TP\times TN-FP\times FN
}{
\sqrt{
(TP+FP)(TP+FN)(TN+FP)(TN+FN)
}}
$$

## False Discovery Rate

$$
FDR=
\frac{FP}
{TP+FP}
$$

---

# 📈 Reported Results

The published study reports that the proposed framework achieved approximately **95% classification accuracy**. One reported result gives approximately **95.077% accuracy and 95.074% precision**, with a reported false discovery rate of about **4.92%**.

The study also evaluates:

* Accuracy
* Precision
* Sensitivity
* Specificity
* F1-score
* MCC
* False prediction parameters
* Computational characteristics

These reported values belong to the experimental setup in the publication and should not be treated as guaranteed performance on unseen field conditions.

---

# 🧪 Ablation Study

Ablation experiments can be used to determine the contribution of individual components.

```text
Full Model
    │
    ├── Remove SA-POA
    ├── Remove CNN Features
    ├── Remove LBP/LGP
    ├── Remove GLCM/Shape
    ├── Remove Residual Connections
    ├── Remove Bi-RNN
    └── Remove SCA
```

The publication reports ablation experiments involving segmentation, SCA, residual Bi-RNN components, and feature groups.

---

# 🆚 Baseline Models

The proposed RBRNN-SCA architecture can be compared against:

* CNN
* RNN
* Bi-RNN
* BRNN
* ResNet-based classifiers
* CNN-RNN hybrids
* Conventional deep learning classifiers
* Transformer-based vision models
* Other plant disease classification methods

For fair evaluation, all models should use the same:

* Dataset
* Train/test partition
* Preprocessing
* Evaluation metrics
* Data augmentation policy

---

# 🔬 Experimental Workflow

```text
                 Dataset
                    │
                    ▼
           Data Pre-processing
                    │
                    ▼
           Leaf Segmentation
                    │
                    ▼
                 SA-POA
                    │
                    ▼
          Segmented Leaf Image
                    │
       ┌────────────┼─────────────┐
       ▼            ▼             ▼
      CNN        LBP + LGP     GLCM + Shape
       │            │             │
       └────────────┼─────────────┘
                    ▼
              Feature Fusion
                    │
                    ▼
              Residual Bi-RNN
                    │
                    ▼
           Spatial Channel
              Attention
                    │
                    ▼
             Classification
                    │
                    ▼
              Evaluation
```

---

# 🌾 Agricultural Applications

The framework can support:

* Precision agriculture
* Smart farming
* Crop health monitoring
* Automated disease diagnosis
* Agricultural mobile applications
* Greenhouse monitoring
* Large-scale crop surveillance
* Drone-based crop inspection
* IoT-enabled agriculture
* Decision-support systems for farmers

---

# 📱 Mobile / Edge Deployment

The framework can potentially be adapted for mobile and edge-based plant disease detection.

```text
Smartphone Camera
       │
       ▼
Plant Leaf Image
       │
       ▼
Pre-processing
       │
       ▼
Leaf Segmentation
       │
       ▼
Feature Extraction
       │
       ▼
RBRNN-SCA
       │
       ▼
Disease Prediction
       │
       ▼
Farmer / Agricultural Expert
```

The publication discusses a mobile/Android-oriented application as a practical implication for image-based disease recognition.

---

# 🌐 IoT-Enabled Smart Agriculture Extension

The model can be integrated into an IoT agriculture architecture:

```text
┌───────────────────────┐
│ Agricultural Field    │
│                       │
│ Plants / Crops        │
└──────────┬────────────┘
           │
           ▼
   Camera / IoT Sensors
           │
           ▼
     Edge Computing
           │
           ▼
  RBRNN-SCA Classifier
           │
      ┌────┴─────┐
      ▼          ▼
 Disease       Healthy
 Detection      Plant
      │
      ▼
   IoT Gateway
      │
      ▼
 Cloud / Dashboard
      │
      ▼
 Farmer / Agronomist
```

---

# 🛠️ Technology Stack

## Programming

* Python 3.x
* NumPy
* Pandas
* OpenCV
* SciPy

## Computer Vision

* OpenCV
* Image preprocessing
* Image segmentation
* Binary thresholding
* Texture analysis
* Shape analysis

## Deep Learning

* TensorFlow / Keras
* PyTorch
* CNN
* Bi-RNN
* Residual learning
* Attention mechanisms

## Feature Extraction

* CNN
* Local Binary Pattern
* Local Gradient Pattern
* Gray-Level Co-occurrence Matrix
* Shape descriptors

## Optimization

* Self-Adaptive Pelican Optimization Algorithm
* Threshold optimization
* Hidden-neuron optimization

---

# 📁 Suggested Project Structure

```text
plant-leaf-disease-rbrnn-sca/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   ├── contrast_enhancement.py
│   │   └── color_transform.py
│   │
│   ├── segmentation/
│   │   ├── thresholding.py
│   │   └── sa_poa.py
│   │
│   ├── features/
│   │   ├── cnn_features.py
│   │   ├── lbp.py
│   │   ├── lgp.py
│   │   ├── glcm.py
│   │   ├── shape_features.py
│   │   └── feature_fusion.py
│   │
│   ├── models/
│   │   ├── birnn.py
│   │   ├── residual_birnn.py
│   │   ├── spatial_channel_attention.py
│   │   └── rbrnn_sca.py
│   │
│   ├── optimization/
│   │   └── sa_poa_optimizer.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validation.py
│   │   └── early_stopping.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── confusion_matrix.py
│       └── visualization.py
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── segmentation.ipynb
│   ├── feature_extraction.ipynb
│   └── model_evaluation.ipynb
│
├── experiments/
│   ├── baseline/
│   ├── ablation/
│   └── rbrnn_sca/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrices/
│   └── models/
│
└── tests/
    ├── test_segmentation.py
    ├── test_features.py
    └── test_model.py
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/plant-leaf-disease-rbrnn-sca.git
cd plant-leaf-disease-rbrnn-sca
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Example Requirements

```text
numpy
pandas
opencv-python
scipy
scikit-image
scikit-learn
matplotlib
seaborn
tensorflow
keras
torch
torchvision
jupyter
```

---

# ▶️ Example Execution

### Preprocess images

```bash
python src/preprocessing/image_preprocessing.py
```

### Run SA-POA segmentation

```bash
python src/segmentation/sa_poa.py
```

### Extract CNN features

```bash
python src/features/cnn_features.py
```

### Extract handcrafted features

```bash
python src/features/lbp.py
python src/features/lgp.py
python src/features/glcm.py
python src/features/shape_features.py
```

### Fuse features

```bash
python src/features/feature_fusion.py
```

### Train RBRNN-SCA

```bash
python src/models/rbrnn_sca.py
```

### Run complete experiment

```bash
python experiments/rbrnn_sca/run_experiment.py
```

### Evaluate model

```bash
python src/evaluation/metrics.py
```

---

# 🔬 Research Contributions

The framework integrates several complementary techniques:

### 1. Adaptive Leaf Segmentation

SA-POA is used to optimize the binary segmentation threshold.

### 2. Multi-Level Feature Extraction

CNN, LBP, LGP, GLCM, and shape descriptors are combined.

### 3. Feature Fusion

The three feature groups are fused to create a richer representation.

### 4. Residual Bi-RNN

Residual connections are integrated with bidirectional recurrent learning.

### 5. Spatial Channel Attention

SCA emphasizes informative spatial and channel-level representations.

### 6. Parameter Optimization

SA-POA is used to optimize segmentation and selected RBRNN parameters.

### 7. Automated Disease Classification

The final RBRNN-SCA model predicts the plant disease class.

These components form the principal architecture described in the published research.

---

# 🚀 Future Research Directions

Potential extensions include:

### 1. Field Image Generalization

Train and evaluate using images captured under:

* Different lighting
* Shadows
* Rain
* Dust
* Occlusion
* Complex backgrounds

### 2. Lightweight Edge AI

Develop compressed RBRNN-SCA models for:

* Smartphones
* Raspberry Pi
* NVIDIA Jetson
* Edge AI cameras
* Agricultural robots

### 3. Explainable AI

Integrate:

* Grad-CAM
* SHAP
* LIME
* Integrated Gradients

to explain disease predictions.

### 4. Disease Severity Estimation

Extend classification into:

```text
Healthy
   ↓
Early Disease
   ↓
Moderate Disease
   ↓
Severe Disease
```

### 5. Multi-Disease Detection

Allow multiple diseases to be identified simultaneously.

### 6. Drone-Based Detection

Integrate the model with UAV imagery for large agricultural fields.

### 7. IoT-Based Continuous Monitoring

Combine:

```text
Camera
  +
Environmental Sensors
  +
Edge AI
  +
IoT Gateway
  +
Cloud Analytics
```

### 8. Multimodal Plant Intelligence

Future systems could combine leaf images with:

* Soil moisture
* Temperature
* Humidity
* Weather
* Nutrient information
* Crop growth stage

for comprehensive crop-health intelligence.

---

# 📚 Publication

**Manogaran, N.; Shankar, Y. B.; Raja, R.; Jayakumar, A. K.; Murugan, S.; Alkhayyat, A.; Jain, S.**

**“An automated plant leaf disease classification framework using three-step feature extraction-based residual Bi-RNN with spatial channel attention.”**

*Discover Computing*, **29**, Article 172, 2026.

**DOI:** `10.1007/s10791-025-09826-5`

**Published:** 24 March 2026.

---

# 📖 Citation

```bibtex
@article{manogaran2026automated,
  title={An automated plant leaf disease classification framework using
         three-step feature extraction-based residual Bi-RNN with
         spatial channel attention},
  author={Manogaran, Nalini and
          Shankar, Yamini Bhavani and
          Raja, Rajalakshmi and
          Jayakumar, Aarav Kannan and
          Murugan, Shanmuganathan and
          Alkhayyat, Ahmad and
          Jain, Shitanshu},
  journal={Discover Computing},
  volume={29},
  pages={172},
  year={2026},
  doi={10.1007/s10791-025-09826-5}
}
```

---

# 👩‍🔬 Research Profile

**R. Rajalakshmi**
Department of Computer Science and Engineering
Sathyabama Institute of Science and Technology
Chennai, Tamil Nadu, India

### Research Areas

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Deep Learning
* Plant Disease Detection
* Agricultural AI
* Explainable AI
* Edge AI
* IoT
* Smart Agriculture
* Neural Networks
* Intelligent Image Processing

---

# 🔗 Research Pipeline Summary

```text
              PLANT LEAF IMAGE
                     │
                     ▼
            IMAGE PREPROCESSING
                     │
                     ▼
          SA-POA OPTIMIZED
            SEGMENTATION
                     │
                     ▼
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
       CNN       LBP + LGP    GLCM + Shape
        │            │            │
        └────────────┼────────────┘
                     ▼
              FEATURE FUSION
                     │
                     ▼
              RESIDUAL Bi-RNN
                     │
                     ▼
          SPATIAL CHANNEL
              ATTENTION
                     │
                     ▼
               SOFTMAX
                     │
                     ▼
          PLANT DISEASE CLASS
```

