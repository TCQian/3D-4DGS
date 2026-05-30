# Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation
Seungjun Oh, Younggeun Lee, Hyejin Jeon, Eunbyung Park

### [[Paper](https://arxiv.org/abs/2505.13215)] [[Project Page](https://ohsngjun.github.io/3D-4DGS/)]

![main](assets/main.jpg)


## 🔥 Overview

**3D-4D Gaussian Splatting** introduces a hybrid representation that combines 3D and 4D Gaussians to model dynamic scenes efficiently — reducing memory, improving speed, and preserving quality.

## 📌 Key Features

- Efficient hybrid representation (3D for static, 4D for dynamic)
- Faster training than 4DGS with similar or better quality
- Drop-in replacement for existing 4DGS pipelines


## 📦 Installation

```shell
git clone https://github.com/ohsngjun/3D-4DGS.git
cd 3D-4DGS
conda env create --file environment.yml
conda activate 3d4dgs
```

## 📁 Data preparation
### Neural 3D Video Dataset
Download the dataset [here](https://github.com/facebookresearch/Neural_3D_Video).
After downloading the data, preprocess it using:
```shell
python scripts/n3v2blender.py $path_to_dataset
```

## 🏃‍♂️ Training
Single sequence training:
```shell
python main.py --config configs/n3v/default.yaml --model_path <model save path> --source_path <dataset path>
```
Train all sequences:
```shell
bash train.sh
```
Don't forget to adjust dataset paths in train.sh.

## 🧪 Testing / Evaluation

```shell
python main.py --config configs/n3v/default.yaml --model_path <model path> --source_path <dataset path> --start_checkpoint <model_path>/chkpnt6000.pth --val
```

## 🙏 Acknowledgement
This project builds upon:
- [Real-time 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Ex4DGS](https://github.com/juno181/Ex4DGS)
- [4D-Rotor Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) (data preprocessing)
- [@sorceressyidi](https://github.com/sorceressyidi) (visualization code)

## 📚 Bibtex

```
@article{oh2025hybrid,
  title={Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation},
  author={Oh, Seungjun and Lee, Younggeun and Jeon, Hyejin and Park, Eunbyung},
  journal={arXiv preprint arXiv:2505.13215},
  year={2025}
}
```

## SOC Cluster installation steps
Run these in a GPU instance cluster:
```
salloc -G h100-47
```

Then run these installation:
```
conda create -n 3d4dgs python=3.10 -y
conda activate 3d4dgs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
export CUDA_HOME=/usr/local/cuda-12.6 && export PATH=$CUDA_HOME/bin:$PATH
pip install ./simple-knn/ --no-build-isolation (must insert `#include <float.h>` in `simple-knn/simple_knn.cu`)
pip install ./pointops2/ --no-build-isolation
pip install tqdm mediapy torchmetrics imagesize kornia omegaconf lpips opencv-python
```

I have colmap built, so i can use colmap like this:
```
export PATH=/home/e/e0407638/miniconda3/envs/colmap/bin:$PATH
colmap
```