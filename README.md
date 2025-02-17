# Overview

 <p align="center">
  <img src="figs\Cover.png" alt="The project workflow" title="The project workflow" width="500" />
</p>


This project focuses on applying signal processing techniques to classify human activities based on raw sensor signals from smartphone accelerometers and gyroscopes. The dataset used in this project is sourced from the UCL Machine Learning Repository, comprising measurements from 30 individuals aged 19 to 48. The measurements were collected using smartphones placed on the waist, capturing six different activities:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying


# Data Description

The dataset has undergone preprocessing steps, including noise filtering and segmentation into fixed-width windows of 2.56 seconds with 50% overlap. Each signal window contains 128 samples, sampling frequency of 50 Hz, contributing to the comprehensive nature of the dataset.

# Signal Components

The smartphone measures the following components per measurement:

1. Three-axial linear body acceleration
2. Three-axial linear gravitational acceleration
3. Three-axial angular velocity

For each measurement, there are a total of nine components contributing to the signal.

# Dataset Source

The dataset can be accessed on the UCL Machine Learning Repository: [Human Activity Recognition using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

# Feature Extraction using DSP Techniques

In this project, Digital Signal Processing (DSP) techniques were employed to extract meaningful features from the raw sensor signals. The application of DSP enhances the representation of the data, capturing relevant patterns and characteristics for improved model performance. The DSP techniques used are:
1. **Fast Fourier transformation (FFT)**
2. **Power spectral density (PSD)**
3. **Autocorrelation**


 <p align="center">
  <img src="figs\Feature extraction.png" alt="Picking peaks in DSP signals as features" title="Picking peaks in DSP signals as features" width="500" />
</p>



# Model Development

A Random Forest model was developed from scratch to classify the human activities based on the extracted features. The model leverages the ensemble learning technique, combining multiple decision trees to enhance accuracy and robustness in activity classification.

# Acknowledgements

This project was the wrap up project of the HarvardX course [Using Python for Research](https://www.edx.org/learn/python/harvard-university-using-python-for-research), where i could apply all the skills and knowledge i have learned throughout the course to this project. 
