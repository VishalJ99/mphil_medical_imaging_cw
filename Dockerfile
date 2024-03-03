FROM continuumio/miniconda3

RUN mkdir -p mphil_medical_imaging_cw

COPY . /mphil_medical_imaging_cw
WORKDIR /mphil_medical_imaging_cw

RUN conda env update --file environment.yml

RUN echo "conda activate mphil_medical_imaging_cw" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
