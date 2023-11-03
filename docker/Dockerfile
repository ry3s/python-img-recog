FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    python-is-python3 \
    python3-venv \
    sudo \
    vim \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PYTHONNOUSERSITE=1
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ARG USERNAME=devuser
ARG UID=1234
ARG GID=1234
RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
