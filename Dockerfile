FROM nvcr.io/nvidia/pytorch:23.07-py3

ARG USERNAME=kaggler
ARG GROUPNAME=kaggler
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME
USER kaggler

RUN python -m pip install --upgrade pip
RUN pip install python-dotenv
RUN pip install transformers==4.28.1
RUN pip install wandb==0.15.5
RUN pip install datasets==2.1.0
RUN pip install sentencepiece==0.1.99
RUN pip install tokenizers==0.13.3
RUN pip install openai==0.27.10
RUN pip install langchain==0.0.305
RUN pip install lightgbm==4.1.0
RUN pip install seaborn==0.13.0
RUN pip install catboost==1.2.2