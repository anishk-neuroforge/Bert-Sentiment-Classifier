<div align="center">

# 🚀 BERT Sentiment Classifier

### Production-ready multilingual sentiment analysis with FastAPI, ONNX Runtime, MLflow, and Docker

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)]()
[![License](https://img.shields.io/badge/License-MIT-success.svg)]()

</div>

---

## Overview

A production-style sentiment analysis API powered by a multilingual BERT model.

The project provides:

- REST API built with FastAPI
- Real-time sentiment prediction
- Batch inference
- ONNX Runtime acceleration
- MLflow experiment tracking
- Docker deployment
- Interactive browser dashboard
- Automated API testing

Instead of being just another machine learning notebook, this repository demonstrates how to package a deep learning model into a deployable service.

---

# Features

- 5-Class Sentiment Classification
- FastAPI REST API
- ONNX Optimized Inference
- MLflow Experiment Tracking
- Interactive Dashboard
- Batch Predictions
- Docker Support
- OpenAPI Documentation
- Pytest Test Suite
- Production Folder Structure

---

# Sentiment Classes

| Label | Meaning |
|--------|----------|
| ⭐ | Very Negative |
| ⭐⭐ | Negative |
| ⭐⭐⭐ | Neutral |
| ⭐⭐⭐⭐ | Positive |
| ⭐⭐⭐⭐⭐ | Very Positive |

---

# Tech Stack

| Category | Technologies |
|-----------|--------------|
| Backend | FastAPI |
| Deep Learning | PyTorch |
| Model | BERT |
| NLP | Transformers |
| Inference | ONNX Runtime |
| Experiment Tracking | MLflow |
| Testing | Pytest |
| Deployment | Docker |
| Documentation | Swagger UI |

---

# Architecture

```
               Client

                  │

                  ▼

          FastAPI REST API

      /predict
      /predict/batch
      /health

                  │

        ┌─────────┴─────────┐
        │                   │

    PyTorch Backend    ONNX Runtime

                  │

                  ▼

      BERT Sentiment Classifier

                  │

                  ▼

      Very Negative
      Negative
      Neutral
      Positive
      Very Positive
```

---

# Project Structure

```
bert-sentiment-classifier/

├── app/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── static/
│
├── scripts/
│   └── export_onnx.py
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Quick Start

### Clone

```bash
git clone https://github.com/yourusername/bert-sentiment-classifier.git

cd bert-sentiment-classifier
```

### Install

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload
```

Server

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

Dashboard

```
http://localhost:8000
```

---

# API Example

### Request

```json
{
    "text":"This product exceeded all my expectations."
}
```

### Response

```json
{
  "label":"Very Positive",
  "confidence":0.97,
  "latency_ms":41.8
}
```

---

# Batch Prediction

```json
{
  "texts":[
      "Amazing product",
      "Worst purchase ever",
      "It is okay"
  ]
}
```

---

# ONNX Acceleration

Export the model

```bash
python scripts/export_onnx.py
```

Run with ONNX Runtime

```bash
MODEL_BACKEND=onnx

uvicorn app.main:app
```

---

# MLflow

Launch the tracking UI

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

Track

- Latency
- Prediction
- Backend
- Input Length
- Runtime

---

# Docker

Build

```bash
docker compose up --build
```

---

# Testing

```bash
pytest
```

---

# Performance

- FastAPI asynchronous API
- ONNX optimized inference
- Batch prediction support
- Lightweight deployment
- Production-ready architecture

---

# Roadmap

- Authentication
- GPU inference
- Model versioning
- Prometheus metrics
- Kubernetes deployment
- CI/CD pipeline
- React Dashboard
- User Management
- API Keys

---

# Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

3. Commit your changes

4. Open a Pull Request

---

# License

This project is released under the MIT License.

---

<div align="center">

Built with FastAPI • Transformers • PyTorch • ONNX Runtime

</div>
