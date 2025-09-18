# anomalieDetectionDL
anomalieDetectionDL makes use of different types of autoencoders. These autoencoders are used to extract features from acoustic emissions and generate a compressed representation format. The network is designed to analyse the behavior of anomalies in acoustic data

# Table of content
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## About
Acoustic emission analysis has emerged as a crucial method for wire break detection, especially in the context of monitoring prestressed structures. In this work, the focus is on the use of sensor data reflecting acoustic emissions from the operation of a real wind turbine. The main objective is to detect wire breaks on external tendons that may occur during operation. For this purpose, an approach based on a deep neural network for anomaly detection is investigated.

## Installation
```bash
git clone https://github.com/mojamil96/anomalieDetectionDL.git
cd repo
pip install -r requirements.txt
```

## Usage
* Add the path of the required noise (acoustic emission to be analysed)
* Add the path for the wirebreaks to be superpositioned with the original noise for training purposes
* Choose the type of the model by indicating the 'model_ID'
* Choose how many epochs the model should be trained
* Data is fed to the model in two different types: timeseries and frequenz series
* If a trained model already exists, add the corresponding path in main.py

## Features
| Model ID   | Description       |
|------------|-------------------|
| model_ID_1 | 1024 -> 512 -> 256 AE 256 -> 512 -> 1024 |
| model_ID_3 | 768 -> 512 -> 256 AE 256 -> 512 -> 768 |
| model_ID_4 | 1024 -> 165 -> AE -> 165 -> 1024 timeseries |
| model_ID_5 | 128 LSTM timeseries |
| model_ID_6 | 1 hidden layer with 32 units |
| model_ID_7 | 2 hidden layers with 32-64 units |
