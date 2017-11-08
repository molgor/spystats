# Docker File for Biospytial 3
## This file is intented to build a Docker container image with the necessary dependencies to run Biospytial under a Python3 environment.
## As such the project is a continuous integration given the complexity in dependencies required by Biospytial.

FROM debian
WORKDIR /root
RUN mkdir container
COPY requirements.yml  container_config/ 
COPY .bashrc /root

RUN rm /bin/sh && ln -s /bin/bash /bin/sh


RUN apt-get update && apt-get install -y \
	apt-utils \
	gdal-bin \
	libgdal-dev \
	python3-gdal \
	python3-pip


RUN pip3 install virtualenv
RUN virtualenv biospytial3

RUN source biospytial3/bin/activate && pip3 install -r container_config/requirements.yml


LABEL maintainer="Juan Escamilla Molgora"
#<j.escamillamolgora@lancaster.ac.uk> (@molgor)
#GitRepo: https://github.com/molgor/spystats
LABEL creation_date="07/11/2017"

CMD bash /root/.bashrc
#ENTRYPOINT  bash 