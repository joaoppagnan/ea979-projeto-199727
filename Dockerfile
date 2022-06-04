# import PyTorch ready docker image
FROM cnstark/pytorch:1.10.2-py3.9.12-cuda11.3.1-ubuntu20.04

# setup docker folder
WORKDIR /ea979-projeto

# copy the requirements.txt to the docker folder
COPY requirements.txt /ea979-projeto

# fix the time zone error
RUN apt update
ENV TZ=America/Sao_Paulo
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# download and install the necessary python packages
RUN apt install python3.9 -y
RUN python3.9 -m pip install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# latex font in matplotlib images
RUN apt install texlive dvipng texlive-latex-extra texlive-fonts-recommended ghostscript cm-super -y

# to be able to change the theme of jupyter lab
RUN apt install nodejs -y

# enable a port for jupyter
EXPOSE 8888