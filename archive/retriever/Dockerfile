FROM ubuntu:latest
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y git wget apt-utils vim
RUN mkdir -p /usr/src/fox

COPY . /usr/src/fox/

# ------------------------------------
# Title: MeCab installation
# OS   : MacOS
# ------------------------------------

# required to install mecab on mac & to run konlpy
RUN apt-get install -y openjdk-8-jdk && \
    apt-get install -y libmecab-dev

# install mecab & mecab-ko-dic & mecab-python
RUN cd /usr/src/fox/docs/mecab && \
    tar xzvf mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure --build=aarch64-unknown-linux-gnu && \
    make && \
    make check && \
    make install && \
    ldconfig
RUN cd /usr/src/fox/docs/mecab && \
    tar zxvf mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./configure --build=aarch64-unknown-linux-gnu --with-mecab-config=/usr/local/bin/mecab-config && \
    make && \
    make install
RUN cd /usr/src/fox/docs/mecab && \
    tar zxfv mecab-python-0.996.tar.gz && \
    cd mecab-python-0.996 && \
    python3 setup.py build && \
    python3 setup.py install