# Official CUDA image with NVCC included
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Update base packages and install necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Default working directory
WORKDIR /workspace

CMD ["/bin/bash"]
