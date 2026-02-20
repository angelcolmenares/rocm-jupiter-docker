# 🚀 ROCm Jupyter Lab with Docker

## 📋 Overview

This repository provides a complete Docker-based environment for running Jupyter Lab with ROCm GPU acceleration on AMD GPUs. It's specifically designed for AI/ML development, research, and education using AMD's ROCm platform with PyTorch.

## 🎯 Purpose

Create a production-ready, reproducible environment where you can:

    Experiment with ROCm and PyTorch on AMD GPUs

    Develop AI/ML models with full GPU acceleration

    Share reproducible environments with colleagues

    Learn ROCm programming without complex setup

    Utilize massive VRAM (up to 68GB+ on supported hardware)

## ✨ Features

    ✅ ROCm GPU Acceleration - Full AMD GPU support via Docker

    ✅ Jupyter Lab - Modern web-based interactive development

    ✅ Persistent Workspace - Notebooks and data survive container restarts

    ✅ Network Access - Accessible from any device on your LAN

    ✅ Security - Token-based authentication

    ✅ Easy Management - Docker Compose for simple control

    ✅ Pre-configured - Optimized for ROCm with 68GB+ VRAM systems

    ✅ Template Structure - Organized workspace for projects

## 🖥️ Supported Hardware

    AMD GPUs with ROCm support (Radeon RX, Radeon Pro, Instinct series)

    Tested on: Gamebox AI Max+ with 68GB VRAM

    System: Ubuntu 25.10 or compatible Linux distributions

    Docker with GPU passthrough support

## Install Docker
curl -sSL https://get.docker.com/ | sh

sudo docker info

sudo docker images



## Clone or extract this repository
cd ~

mkdir rocm-jupyter-docker

cd rocm-jupyter-docker

## Create you .env file:
```text
JUPYTER_TOKEN=************************
JUPYTER_PORT=8888
# User configuration
UID=your_user_id
GID=1000
USERNAME=your_user_name
# Try this for stability
HSA_OVERRIDE_GFX_VERSION:11.0.0
# PyTorch memory optimization
# expandable_segments:True
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
# expandable_segments:True
PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
# Enable TF32 for faster math (if supported)
NVIDIA_TF32_OVERRIDE=1
# ROCm optimization
HIP_VISIBLE_DEVICES=0
ROCR_VISIBLE_DEVICES=0
HSA_ENABLE_SDMA=1
# CPU optimization (32 cores)
OMP_NUM_THREADS=16
MKL_NUM_THREADS=16
NUMEXPR_NUM_THREADS=16
# Python optimization
PYTHONUNBUFFERED=1
PYTHONHASHSEED=0
```


## Directory Structure:
- `notebooks/` - Jupyter notebooks
- `datasets/` - Training/testing datasets
- `models/` - Trained models and weights
- `logs/` - Training logs, TensorBoard logs
- `checkpoints/` - Model checkpoints during training
- `experiments/` - Experimental scripts and code
- `results/` - Final results, visualizations, reports

## 🐳 Docker Commands

### Start services (background)
docker compose up -d

### Stop services
docker compose down

### View logs
docker compose logs -f

### Rebuild image
docker compose up --build -d

### Enter container shell
docker exec -it rocm7.2-pytorch-jupyter bash

### Check GPU status
docker exec rocm7.2-pytorch-jupyter rocm-smi

### Access:
- Launch docker container in your terminal:  docker compose up -d
- Jupyter Lab in your browser: http://YOUR-IP:8888


## 🤝 Contributing

Feel free to:

    Report issues with ROCm/Jupyter compatibility

    Suggest improvements for large VRAM utilization

    Add examples of working ROCm models

    Share performance benchmarks

## 📄 License

This project is provided as-is for educational and research purposes. 

## 🙏 Acknowledgements

    AMD for ROCm platform

    PyTorch Team for ROCm support

    Jupyter Project for the excellent notebook interface

    Docker Community for containerization tools

## 🎮 Ready to Code?

Your 68GB VRAM ROCm Jupyter environment is ready! Start with:

    Test your GPU: Run the benchmark notebook

    Try ROCm ResNet: Import from rocm_resnet.py

    Experiment: Use large batch sizes and datasets

    Share: Access from any device on your network

## Happy coding with ROCm! 🚀

Note: This setup is specifically optimized for AMD GPU systems with ROCm support. For NVIDIA GPUs, consider using nvidia/cuda base images instead.

