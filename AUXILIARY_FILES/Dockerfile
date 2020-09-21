# The image corresponding to this dosckerfile can be obtained by 
#docker pull us.gcr.io/broad-dsde-methods/pyro_matplotlib:1.3.0

# This Dockerfile is inspired by https://github.com/anibali/docker-pytorch/blob/master/cuda-10.1/Dockerfile
FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    software-properties-common \
    nmap \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app 
WORKDIR /app

COPY . .


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install conda-build
RUN /home/user/miniconda/bin/conda install conda-build

# Update conda and conda-build
RUN /home/user/miniconda/bin/conda update -n base -c defaults conda \
 && /home/user/miniconda/bin/conda update conda-build \
 && /home/user/miniconda/bin/conda clean -ya

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py385 python=3.8.5 \ 
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py385
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.1-specific steps
RUN conda install -y pytorch==1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch \
 && conda clean -ya

# Install pyro-ppl from PIP
RUN yes | pip install pyro-ppl==1.4.0
# && sudo rm -rf ~/.cache/pip

# Install matplotlib from conda-forge
RUN conda install -y -c conda-forge matplotlib \
 && conda clean -ya

# Install skimage
RUN conda install -y -c conda-forge scikit-image \
 && conda clean -ya

# Upgrade Pillow
RUN yes | pip install --upgrade Pillow

# Install dill which is a better version of pickle
RUN yes | pip install dill==0.3.2  

# Install neptune-client
RUN yes | pip install neptune-client

# Uncomment if you need to run gsutil command from inside the command section of WDL
#RUN curl -sSL https://sdk.cloud.google.com | bash
