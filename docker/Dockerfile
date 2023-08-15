FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo poppler-utils curl git gcc vim wget unzip\
                              lsb-core software-properties-common

# Setup user
RUN useradd --uid 1000 --shell /bin/bash appliedml42
RUN usermod -aG sudo appliedml42
RUN echo "appliedml42 ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
ADD . /app
RUN chown -R appliedml42 /app
ENV HOME /app
WORKDIR /app
USER appliedml42
ENV PATH /app/.local/bin:$PATH

# Install other requirements
RUN pip install --trusted-host pypi.python.org -r ./requirements.txt

# Install lm_dataformat
RUN git clone https://github.com/leogao2/lm_dataformat.git
WORKDIR lm_dataformat
RUN python setup.py install --user
WORKDIR /app

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

# Install Pytorch Nightly
RUN sudo pip uninstall torch torchvision torchaudio -y
RUN sudo pip uninstall torch -y
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
