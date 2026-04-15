# GPU image — includes CUDA 12.2 + cuDNN 8 + TensorFlow 2.15
# TF auto-detects GPU at runtime; falls back to CPU if no GPU present
FROM tensorflow/tensorflow:2.15.0-gpu

LABEL maintainer="AI & Big Data Vocational Course"
LABEL description="Dev container — Skin Cancer Classifier (Keras/TensorFlow + NVIDIA GPU)"

# System deps: image processing + VS Code dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Non-root user expected by VS Code Dev Containers
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

WORKDIR /workspace

# Install Python dependencies as root (available to all users)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Lab config — no token/password for local dev
# Write config for both root and vscode user
RUN mkdir -p /root/.jupyter /home/vscode/.jupyter && \
    printf 'c.ServerApp.token = ""\nc.ServerApp.password = ""\nc.ServerApp.open_browser = False\n' \
    | tee /root/.jupyter/jupyter_server_config.py \
        /home/vscode/.jupyter/jupyter_server_config.py > /dev/null && \
    chown -R vscode:vscode /home/vscode/.jupyter

EXPOSE 8888

USER ${USERNAME}
