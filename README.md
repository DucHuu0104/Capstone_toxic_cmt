---
title: Toxic Comments Classification
emoji: 😻
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
short_description: Demo toxic comment classification
---
📋 Overview
This project focuses on solving a Natural Language Processing (NLP) problem to automatically detect negative or toxic comments on social media platforms such as Facebook, YouTube, or TikTok.
This method is design to address Multi-label Classification: Identify specific types of toxicity (toxic, serve_toxic, obscene, threat, insult, identity_hate)

🗂️ Dataset
The project utilizes the following data sources:

Public Data: The Kaggle - Jigsaw Toxic Comment Classification Challenge dataset (multi-label).

🚀 Pipeline & Methodology

Toxic Comment Multi-Label Classification System:

Lightweight & Efficient MLP Architecture
Author: [Duc Dang Nguyen Huu] Tech Stack: Python, PyTorch, Scikit-learn, Pandas, Unicodedata, Re, Torch, Torch.nn, Pickle,  Gradio Matplotlib, Seaborn, Sklearn, Numpy

1. Project Overview
In the era of exploding unstructured text data on social platforms, detecting toxic content requires a solution that is not only accurate but also possesses extremely fast inference speeds.

This project implements a Multi-label Classification System to detect 6 distinct types of toxicity in text: toxic, serve_toxic, obscene, threat, insult, identity_hate

Instead of jumping straight to heavy, resource-intensive Transformer models (like BERT or RoBERTa), I architected a Hybrid approach: TF-IDF + Neural Network (MLP). This serves as a "Strong Baseline"—optimizing the trade-off between accuracy and computational resources, making it highly suitable for production environments requiring low latency.

2. System ArchitectureThe data processing pipeline is rigorously designed across three stages:A. Preprocessing & CleaningRaw text data contains significant noise. I implemented a strict cleaning protocol:Normalization: Unicode normalization (NFKC) to handle special characters consistently.Regex Filtering: Removal of URLs and non-alphanumeric characters, and reduction of repeated characters (e.g., "goood" -> "good").Lowercasing: Reduces dimensionality and standardizes the vocabulary.B. Feature Engineering (TF-IDF)I utilized TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization:Max Features: 10,000 (Retaining only the most significant tokens to reduce noise).N-gram range (1, 2): Captures both unigrams and bi-grams to preserve local context (e.g., distinguishing "not good" from "good").Stop words removal: Filters out common English words that contribute little to classification logic.C. Model Design (PyTorch MLP)The model is a streamlined Multi-Layer Perceptron optimized for performance:Input Layer: 10,000 dimensions (Matching the TF-IDF vector space).Hidden Layer: 512 units + ReLU Activation (Introducing non-linearity).Regularization: Dropout (0.3) is applied to prevent overfitting, ensuring better generalization on unseen data.Output Layer: 6 units (one for each label). Crucially, I used Logits (no Softmax) combined with BCEWithLogitsLoss to handle independent multi-label probabilities effectively.

3. Training Strategy
Loss Function: BCEWithLogitsLoss (Binary Cross Entropy integrated with Sigmoid). This ensures better numerical stability than applying Sigmoid and BCELoss separately.

Optimizer: AdamW (Learning rate: 1e-3). A variant of Adam with decoupled weight decay, providing faster convergence and better regularization than standard Adam.

Checkpointing: The system monitors validation accuracy and only saves model weights when performance improves.

4. Evaluation & Metrics
The system is evaluated comprehensively using multiple metrics, acknowledging the class imbalance inherent in toxicity datasets:

F1-Score (Micro & Macro): Balances Precision and Recall across all classes.

ROC-AUC Score: Measures the model's ability to distinguish between toxic and non-toxic classes.

Confusion Matrix: Visualizes False Positives vs. False Negatives.

Engineering Note: The codebase includes automated plotting for ROC Curves and Precision-Recall Curves for each specific label. This allows for fine-tuning decision thresholds based on business requirements (e.g., prioritizing Recall for threat detection to minimize missed threats).

💻 Installation & Usage

To run this project locally for development or testing, follow these steps:

1. Clone the repository
"git clone https://huggingface.co/spaces/YOUR_USERNAME/Toxic-Comment-Classification cd Toxic-Comment-Classification"

2. Install dependencies Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.
"pip install -r requirements.txt"

3. Run the application
"python app.py"

Another way to run this project
Online: https://huggingface.co/spaces/Duc0104/toxic_comments_classification



📝 Author
[Duc Nguyen Huu Dang]

Capstone Project