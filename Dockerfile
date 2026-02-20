FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

# Create a non-root user with your UID/GID (will be overridden at runtime)
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=1000

# Create user if it doesn't exist
RUN set -eux; \
    # Check if group exists
    if getent group ${NB_GID} > /dev/null; then \
        existing_group_name=$(getent group ${NB_GID} | cut -d: -f1); \
        if [ "${existing_group_name}" != "${NB_USER}" ]; then \
            groupmod -n ${NB_USER} ${existing_group_name}; \
        fi; \
    else \
        groupadd -g ${NB_GID} ${NB_USER}; \
    fi; \
    # Check if user exists
    if id -u ${NB_UID} > /dev/null 2>&1; then \
        existing_user_name=$(id -nu ${NB_UID}); \
        if [ "${existing_user_name}" != "${NB_USER}" ]; then \
            usermod -l ${NB_USER} -g ${NB_GID} -d /home/${NB_USER} -m ${existing_user_name}; \
        fi; \
    else \
        useradd -l -u ${NB_UID} -g ${NB_GID} -m ${NB_USER}; \
    fi; \
    # Ensure home directory exists and has correct permissions
    mkdir -p /home/${NB_USER}; \
    chown -R ${NB_UID}:${NB_GID} /home/${NB_USER}; \
    # Add user to sudoers (optional)
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${NB_USER}


# System dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    curl \
    htop \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python ML ecosystem
RUN pip install --no-cache-dir \
    # Jupyter
    jupyterlab \
    notebook \
    ipywidgets \
    jupyter-archive \
    # Data science
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    # Deep learning
    torchvision \
    torchaudio \
    transformers \
    datasets \
    accelerate \
    # Utilities
    tqdm \
    pillow \
    opencv-python-headless \
    tensorboard \
    wandb \
    # Optional
    xformers 

RUN pip install --no-cache-dir "fastprogress==1.0.3"
RUN pip install -Uqq fastai fastbook fastcore
RUN pip install -U ddgs

# Create workspace and set permissions
RUN mkdir -p /workspace && \
    chown -R ${NB_UID}:${NB_GID} /workspace


# Install .NET dependencies
RUN apt-get update && \
    apt-get install -y wget software-properties-common

# Install .NET 10
RUN wget https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-10.0


# Switch to non-root user
USER ${NB_USER}

# Install .NET Interactive
RUN dotnet tool install -g Microsoft.dotnet-interactive

# Add to PATH
ENV PATH="/home/${NB_USER}/.dotnet/tools:${PATH}"

# Install Jupyter kernel
RUN DOTNET_ROLL_FORWARD=Major dotnet interactive jupyter install
ENV DOTNET_ROLL_FORWARD=Major

WORKDIR /workspace

EXPOSE 8888

# Health check
#HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#    CMD python -c "import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(1); result = sock.connect_ex(('localhost', 8888)); sock.close(); exit(result)"
