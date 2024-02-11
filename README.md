# Facial Expression Recognition Project

## Overview

This project applies machine learning techniques to recognize facial expressions from images. It leverages Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for feature reduction and explores classification using Support Vector Machines (SVM) and the Nearest Centroid Classifier. The goal is to efficiently process facial images for accurate expression recognition, highlighting the strengths of each method in handling high-dimensional data.

## Features

- **Data Preprocessing:** Prepare facial image data for analysis.
- **PCA for Dimensionality Reduction:** Reduce the feature space while retaining the most significant information.
- **Reconstruction Error Analysis:** Evaluate the impact of reducing dimensions with PCA on image reconstruction.
- **Image Reconstruction:** Demonstrate how images can be reconstructed using varying numbers of principal components.
- **LDA for Enhanced Class Separability:** Improve the separation between different facial expression classes.
- **Classification Comparison:** Assess and compare the accuracy of SVM and Nearest Centroid Classifier post-feature reduction.

## Requirements

- MATLAB
- Image Processing Toolbox (Optional for some image-related operations)

## Usage

1. **Data Preparation:** Ensure your dataset is organized appropriately, with facial images labeled according to their expression classes.
2. **Run Analysis:** Execute the script step-by-step to perform PCA and LDA, followed by classification. Adjust parameters as needed for your dataset and goals.
