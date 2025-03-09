FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /app/

RUN conda env create -f environment.yml || true

SHELL ["/bin/bash", "-c"]
RUN echo "conda activate PPIOMEGA_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/PPIOMEGA_env/bin:$PATH

RUN conda run -n PPIOMEGA_env pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric

COPY . /app

CMD ["bash"]
