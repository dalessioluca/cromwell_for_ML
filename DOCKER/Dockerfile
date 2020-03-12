# The image corresponding to this dosckerfile can be obtained by 
#docker pull us.gcr.io/broad-dsde-methods/pyro_matplotlib:1.3.0

# This Dockerfile is inspired by https://github.com/anibali/docker-pytorch/blob/master/cuda-10.1/Dockerfile
FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app 
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
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
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.1-specific steps
RUN conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch \
 && conda clean -ya

# Install pyro-ppl from PIP
RUN yes | pip install pyro-ppl==1.3.0
# && sudo rm -rf ~/.cache/pip

# Install jq from conda-forge
RUN conda install -y -c conda-forge jq \
 && conda clean -ya

# Install matplotlib from conda-forge
RUN conda install -y -c conda-forge matplotlib \
 && conda clean -ya

# Install jupyter_contrib_nbextensions from conda-forge
RUN conda install -y -c conda-forge jupyter_contrib_nbextensions \
 && conda clean -ya

# Install moviepy from conda-forge
RUN conda install -y -c conda-forge moviepy \
 && conda clean -ya

# Uncomment if you need to run gsutil command from inside the command section of WDL
#RUN curl -sSL https://sdk.cloud.google.com | bash
