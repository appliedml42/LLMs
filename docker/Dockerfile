FROM nvcr.io/nvidia/pytorch:22.01-py3

# Install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo poppler-utils curl git gcc vim wget unzip\
                              lsb-core software-properties-common llvm-9-dev

## Install NCCL2
RUN apt-get -y remove libnccl2 libnccl-dev
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && sudo apt-get install -y libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4 cmake \
    build-essential pkg-config libgoogle-perftools-dev

# Setup user
RUN useradd --uid 1000 --shell /bin/bash appliedml42
RUN usermod -aG sudo appliedml85
RUN echo "appliedml85 ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
ADD . /app
RUN chown -R appliedml85 /app
ENV HOME /app
WORKDIR /app
USER appliedml85
ENV PATH /app/.local/bin:$PATH

# Install other requirements
RUN pip install --trusted-host pypi.python.org -r ./requirements.txt

# Install lm_dataformat
RUN git clone https://github.com/leogao2/lm_dataformat.git
WORKDIR lm_dataformat
RUN python setup.py install --user
WORKDIR /app

# Install OpenMPI
RUN sudo mv /usr/local/mpi /usr/local/bk_mpi
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
RUN gunzip -c openmpi-4.0.0.tar.gz | tar xf -
WORKDIR openmpi-4.0.0
RUN ./configure --prefix=/usr/local/openmpi
RUN make -j 8 all
RUN sudo make install
ENV LD_LIBRARY_PATH /usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/openmpi/bin:$PATH
WORKDIR /app
RUN rm -rf openmpi*

# Install SPM CLI Tools
RUN git clone https://github.com/google/sentencepiece.git
WORKDIR sentencepiece
RUN mkdir build
WORKDIR build
RUN cmake ..
RUN make -j $(nproc)
RUN sudo make install
RUN sudo ldconfig -v
WORKDIR /app
