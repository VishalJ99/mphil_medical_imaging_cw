FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install -y build-essential python3-dev

COPY Dataset /mphil_medical_imaging_cw/Dataset
COPY src /mphil_medical_imaging_cw/src
COPY configs /mphil_medical_imaging_cw/configs
COPY environment.yml /mphil_medical_imaging_cw

WORKDIR /mphil_medical_imaging_cw

RUN conda env update --file environment.yml

RUN echo "conda activate mphil_medical_imaging_cw" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
