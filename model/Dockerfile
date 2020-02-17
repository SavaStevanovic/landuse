FROM nvcr.io/nvidia/pytorch:20.01-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev && git clone https://github.com/NVIDIA-AI-IOT/torch2trt && pip install -e torch2trt/
ENV QT_X11_NO_MITSHM=1
CMD bash
