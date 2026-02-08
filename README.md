# PS1: ReelSense-Explainable Hybrid Movie Recommendation System with Diversity Optimization

## Overview
ReelSense is a hybrid movie recommender system built on the MovieLens dataset. It combines collaborative filtering, content-based filtering, matrix factorization (SVD), and a novelty/diversity booster to generate accurate, diverse, and explainable movie recommendations.

Unlike traditional recommenders that push only popular movies, ReelSense promotes long-tail discovery while providing clear explanations for every recommendation.

## Key Features
* Hybrid recommendation pipeline (CF + Content + SVD + Novelty)
* Diversity optimization to reduce filter bubbles
* Explainable recommendations with score breakdown
* Multiple model variants tested and compared

## Dataset
**MovieLens Latest Small**

* 100,836 ratings
* 610 users
* 9,742 movies
* Tags + genres used for content features

## Methodology
Final recommendation score:
[
Score(u,i) = \alpha CF + \beta Content + \gamma SVD + \delta Novelty
]

Components:
* Collaborative Filtering (cosine similarity)
* Content Similarity (TF-IDF on genres + tags)
* SVD Matrix Factorization (latent embeddings)
* Novelty boost (popularity penalty)

## Results
Hybrid model improves over SVD baseline:

* Precision@10: **+8%**
* Recall@10: **+13%**
* NDCG@10: **0.52**
* Improved catalog coverage + novelty

## Visualizations Included
* Rating distribution
* Genre frequency chart
* Popularity long-tail plot
* Accuracy vs diversity comparison
* Model metric comparison (Precision/Recall/NDCG)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Surprise (SVD), Matplotlib/Seaborn

---

# PS2: Radiology Report Generation-Automated Radiology Report Generation

## Overview
CXR-Insight is a lightweight radiology report generation model designed to generate structured **Findings** and **Impression** sections from chest X-ray images. The system is inspired by cognitive simulation principles and mimics radiologist reasoning through multi-level perception and verification loops.

## Key Features
* Generates radiology-style structured reports
* Uses hierarchical visual features (pixel, region, organ)
* Triangular attention reasoning loop to reduce hallucination
* GPT-2 cross-attention decoding for image-grounded text generation

## Dataset
**IU-Xray Dataset**

* Used subset: 500 frontal images
* Train: 400
* Validation: 100
* Reports include Findings + Impression

## Architecture (Core Modules)

### PRO-FA (Hierarchical Visual Perception)
Extracts pixel-level, region-level, and organ-level features using ViT.

### MIX-MLP (Disease Hypothesis Module)
Dual-path MLP for disease-aware feature learning (residual + expansion).

### RCTA (Triangular Cognitive Attention)
Attention loop simulating radiologist reasoning:
Image → Context → Hypothesis → Verification

### GPT-2 Cross Attention Decoder
Generates text conditioned on verified image embeddings.

## Results
* Epoch 5 Generation Loss: **0.3054**
* Reports are fluent and clinically formatted
* Strong performance on normal cases (clear lungs, no effusion, no pneumothorax)
* Minor hallucinations observed due to limited training

## Visualizations Included
* Sample generated report vs ground truth
* Training loss curve
* Report Length Distribution
  
## Tech Stack
Python, PyTorch, HuggingFace Transformers (ViT + GPT2), Pandas, PIL
