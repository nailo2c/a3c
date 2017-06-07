# a3c

<sub>使用PyTorch實作a3c演算法，參考了openai/universe-starter-agen以tensorflow實作的版本，以及ikostrikov/pytorch-a3c以PyTorch實作的版本。  
以ikostrikov為主要參考，加上自行修改的一些部分，並以盡量精簡行數、寫出容易理解的code為目標。</sub>

# Dependencies

* Anaconda
* PyTorch

# Getting Started

以下以Ubuntu 16.04 LTS環境為準，安裝Anaconda時請一路Enter與Yes到底。

```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
source .bashrc
conda install pytorch torchvision -c soumith
conda install opencv
conda install libgcc
pip install gym[Atari]
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

# Rendering on a server

如果是跑在server上，需要依靠xvfb創造虛擬畫面支持rendering。

```
xvfb-run -s "-screen 0 1400x900x24" python main.py --env-name "Pong-v0" --num-processes 16
```

# References

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[openai/universe-starter-agen](https://github.com/openai/universe-starter-agent)  
[ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
