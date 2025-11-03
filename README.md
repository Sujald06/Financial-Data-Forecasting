# 📊 Financial Data Forecasting
# 📈 Stock Price Prediction Using LSTM

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?logo=streamlit)](https://stock-price-prediction-lstm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Overview

This project demonstrates **Stock Closing Price Prediction** using **Long Short-Term Memory (LSTM)** neural networks implemented in **Keras** and **TensorFlow**.  
It leverages financial data fetched via the **yfinance** API and provides an **interactive visualization and prediction interface** built with **Streamlit**.

The goal is to deliver a complete **end-to-end machine learning solution** — from data extraction and preprocessing to model training, evaluation, and web deployment.

### 🔍 Key Highlights
- Real-time stock data fetched via `yfinance`
- Sequential modeling with **LSTM layers**
- Model built and trained in **Keras**
- Interactive **Streamlit dashboard** for predictions
- Deployed live on **Streamlit Cloud**

---

## 📂 Folder Structure

```

Financial-Data-Forecasting/
│
├── LSTM Model.ipynb        # Jupyter notebook for training, testing, and model building
├── app.py                  # Streamlit web app
├── keras_model.h5          # Saved trained LSTM model
├── requirements.txt        # List of dependencies for deployment
└── README.md               # Project documentation

````

---

## ⚙️ Setup Instructions

### 🧩 1. Clone the Repository
```bash
git clone https://github.com/Sujald06/Financial-Data-Forecasting.git
cd Financial-Data-Forecasting
````

### 🧩 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # For Windows
# or
source .venv/bin/activate       # For macOS/Linux
```

### 🧩 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, install manually:

```bash
pip install pandas numpy matplotlib pandas_datareader yfinance streamlit tensorflow keras
```

---

## 💡 Usage

### 🧠 Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
```

### 📊 Load Financial Data

```python
yf.pdr_override()
df = pdr.get_data_yahoo(ticker, start_date, end_date)
```

### ▶️ Run the Streamlit App Locally

```bash
streamlit run app.py
```

---

## 🚀 Deployment (Streamlit Cloud)

### Step 1: Version Control with Git

Push your app and `requirements.txt` file to a GitHub repository.

### Step 2: Deploy on Streamlit Cloud

1. Visit [https://share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repo, branch, and app file (e.g., `app.py`)
4. Click **Deploy**

### Step 3: Manage Your App

Streamlit Cloud allows you to:

* Add custom app name and thumbnail
* Manage secrets and credentials
* Monitor logs and performance

---

## 🧰 Tech Stack

| Category        | Tools                                                                     |
| --------------- | ------------------------------------------------------------------------- |
| Language        | Python                                                                    |
| Libraries       | TensorFlow, Keras, pandas, NumPy, matplotlib, yfinance, pandas_datareader |
| Visualization   | Matplotlib, Streamlit                                                     |
| Deployment      | Streamlit Cloud                                                           |
| Version Control | Git, GitHub                                                               |

---

## 📉 Results

* Achieved high accuracy in predicting stock closing prices on test data.
* Interactive app enables users to visualize actual vs predicted trends.
* Easy extension to multiple tickers and different timeframes.

*(Add a sample chart or screenshot here for visual appeal)*

---

## 🧾 Requirements

A `requirements.txt` file should include:

```
pandas
numpy
matplotlib
pandas_datareader
yfinance
streamlit
tensorflow
keras
```

---


---

## 👨‍💻 Author

**Sujal D.**
📧 [GitHub Profile](https://github.com/Sujald06)
💼 Data Science & AI Enthusiast

---

> 💬 *“Predicting the market might be tough, but understanding it is the first step.”*


