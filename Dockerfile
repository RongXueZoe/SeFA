FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget --quiet https://micro.mamba.pm/install.sh -O /tmp/install.sh && /bin/bash /tmp/install.sh && \
    ~/.local/bin/micromamba shell init -s bash -r ~/micromamba

# set path to conda
ENV PATH=/root/.local/bin:$PATH

ENV CUDA_HOME '/usr/local/cuda'
ENV LD_LIBRARY_PATH $CUDA_HOME/lib:$CUDA_HOME/lib64:/lib:/lib
ENV PATH $CUDA_HOME/bin:$PATH

# create conda environment
COPY ./docker_environment.yaml /tmp/docker_environment.yaml
RUN micromamba env create -f /tmp/docker_environment.yaml
RUN echo "micromamba activate sefapolicy" >> ~/.bashrc

WORKDIR /workspace
RUN /root/micromamba/envs/sefapolicy/bin/pip install garage --no-deps

RUN apt-get update && apt-get install -y libosmesa6-dev patchelf && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*