## Udacity CarND Capstone Submission

### Installation

1. **Clone the repository**

The repository can be found [here](https://github.com/CarNDCapstone/CarND-Capstone). To clone the repository, do

```
git clone https://github.com/CarNDCapstone/CarND-Capstone.git
```
Also, make sure you change directories to get inside the repo
```
cd CarND-Capstone
```

2. This setup depends a [Docker](https://www.docker.com/) image. If you'd like to know more about Docker containers, click [here](https://www.docker.com/resources/what-container).

**Why Docker?**
* Makes software portable across platforms, like a virtual machine.
* Zero overhead when run on Linux (VMs can have a lot overhead - definitely I/O, sometimes compute, depending on the [hypervisor](https://en.wikipedia.org/wiki/Hypervisor))
* GPU acceleration - VMs don't allow the user to run CUDA. This is because CUDA in a Linux VM would expect to talk to a native Linux driver, but the host OS could be Linux, Windows or MacOS. By contrast, Docker containers running on Linux hosts can access CUDA-capable GPUs for compute (see the GPU Acceleration section below).


**Docker runtime installation instructions**

* [Ubuntu Linux](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [CentOS Linux](https://docs.docker.com/install/linux/docker-ce/centos/)
* [RHEL Linux](https://docs.docker.com/ee/docker-ee/rhel/)
* [MacOS](https://download.docker.com/mac/stable/Docker.dmg)
* [Windows](https://download.docker.com/win/stable/Docker%20Desktop%20Installer.exe)

**Note:** the authors only tested the setup on Ubuntu Linux and MacOS.

**GPU acceleration:** This code uses neural network detection and classification for the traffic light model. GPU acceleration is only available in this setup when Docker runs natively on Linux. Fortunately, our detection model should be fast enough on a CPU, but using GPU acceleration is recommended whenever possible.

Supported GPUs include: Tesla K80, Tesla M60, Tesla P100, Tesla V100, GTX 1070/1080/1080Ti, RTX 2070/2080/2080Ti. Titan X, Titan XP and Titan RTX should also work.

To get GPU acceleration, after installing Docker, you'll need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). We're copying the nvidia-docker instructions from the [NVIDIA github repo](https://github.com/NVIDIA/nvidia-docker) here.

*nvidia-docker setup on Ubuntu Linux*

```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

*nvidia-docker setup on CentOS 7 / RHEL 7*

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker
```

If you have an NVIDIA GPU and correctly installed Docker and nvidia-docker, running a CUDA container (will be pulled automatically) should show your driver version (here 440.59) and GPU type (here GeForce RTX 207):

```
docker run --rm -it nvidia/cuda nvidia-smi
Sun Mar 29 21:32:42 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.59       Driver Version: 440.59       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2070..  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   40C    P8     9W /  N/A |   1478MiB /  7982MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```

**Docker without sudo:** Normally, starting a Docker container requires `sudo`. This required can be removed by configuring the Docker runtme correctly. On MacOS and Windows, this should be the default. On Linux, one has to issue a few commands

**Memory issues on MacOS:** Docker lets containers use only 2 GB RAM on MacOS by default. Open the Docker app on MacOS, click Settings, and change to at least 4 GB. More is preferable.

2. **Pull Docker container or build from source**

The container is already pre-built and is available on [Dockerhub](https://hub.docker.com/r/mkolod/udacity_carnd_capstone). You can pull the container as follows:
```
docker pull mkolod/udacity_carnd_capstone
```

If you'd rather build the container from source, note that there are two Dockerfiles at the root of the CarND-Capstone repository that we cloned earlier. We want to build using `Dockerfile-modern-gpu`. To do that, issue the following command while at the root of the repo:
```
docker build -t mkolod/udacity_carnd_capstone -f Dockerfile_modern_gpu .
```
This will build a Docker image called `mkolod/udacity/carnd_capstone`. Note that building from source may take a few minutes. Above `-f` specifies the Dockerfile we want to use (by default the build will use a file called `Dockerfile`) and `-t` stands for "tag," i.e. how we want to name the container. Let's name it the same as the one we would have pulled from Dockerhub.

3. **Install the Udacity simulator**

The instructions how to install the simulator on your particular platform can be found [here](https://github.com/udacity/CarND-Capstone/releases/tag/v1.3).

4. **Launch the container**
While still at the root of the CarND-Capstone repository, launch a Docker container
```
docker run --rm -it -p 4567:4567 -v `pwd`:/workspace mkolod/udacity_carnd_capstone
```

The above command creates an ephemeral container, to be removed after we type `exit` in the shell session (`--rm` flag). We want to launch an interactive conatiner with a terminal (`-it` stands for "interactive terminal"). We expose port 4567 since this is the port ROS will use to communicate with the Udacity simulator. Since Docker containers run with their own ephemeral filesystem and we want to mount the repository we just cloned inside the container, we pass the volume argument (`-v` for "volume"). The repository will then be mounted into the container under the `/workspace` path.

When the Docker container launches, the user will normally be root, and the hostname will be some hash. This is why you can expect a prompt like this once the container launches:
```
root@28064d9756ab:/capstone/ros
```

**Launch with GPU support:** If your host OS is Linux, you have an NVIDIA GPU and have a recent NVIDIA driver (we tested with `440.59`), and you installed `nvidia-docker` as described above, you can add the `--gpus all` flag while launching the container. So, the full launch command would look like so:

```
docker run --rm -it -p 4567:4567 -v `pwd`:/workspace --gpus all mkolod/udacity_carnd_capstone
```

5. **Set up ROS environment and launch nodes**

We mounted the cloned repository in Docker under `/workspace`, so let's change directories.

```
cd /workspace/ros
```

Now, let's run `catkin_make` to build our nodes.

```
catkin_make
```

Before we launch the nodes, we need to set up our shell environment, which we can do like so:
```
source devel/setup.bash
```

Now, let's launch our ROS nodes:

```
roslaunch launch/styx.launch
```

You will start getting log output after this command. Note that until you launch the Udacity simulator, you might see warnings such as this one:


```
[ WARN] [1585517476.418286130]: position is missing
[ WARN] [1585517476.418361765]: waypoint is missing
[ WARN] [1585517476.418408371]: velocity is missing

```

The warnings will go away once you launch the simulator.
