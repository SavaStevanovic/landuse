docker build -t pytorch2001playground .
xhost + 
docker run -e DISPLAY=$DISPLAY --ipc=host --gpus all -it -p 6006:6006 -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix pytorch2001playground
