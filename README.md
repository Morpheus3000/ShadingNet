# ShadingNet: Image Intrinsics by Fine-Grained Shading Decomposition

A. S. Baslamisli\*, P Das\*, H. A. Le, S. Karaoglu, T. Gevers [ShadingNet: Image Intrinsics by Fine-Grained Shading Decomposition](https://arxiv.org/abs/1912.04023). <sub><sup>\* denotes equal contribution.</sup></sub>

We provide the tensorflow implementation of "ShadingNet: Image Intrinsics by Fine-Grained Shading Decomposition", IJCV2021. The provided models were tested with 1.10.1.

The pretrained model can be downloaded from [here](https://uvaauas.figshare.com/ndownloader/files/38127018)

In this paper, we propose to decompose the shading component into direct (illumination) and indirect shading (ambient light and shadows) subcomponents. The aim is to distinguish strong photometric effects from reflectance variations. An end-to-end deep convolutional neural network (ShadingNet) is proposed that operates in a fine-to-coarse manner with a specialized fusion and refinement unit exploiting the fine-grained shading model. It is designed to learn specific reflectance cues separated from specific photometric effects to analyze the disentanglement capability. A large-scale dataset of scene-level synthetic images of outdoor natural environments is provided with fine-grained intrinsic image ground-truths. Large scale experiments show that our approach using fine-grained shading decompositions outperforms state-of-the-art algorithms utilizing unified shading on NED, MPI Sintel, GTA V, IIW, MIT Intrinsic Images, 3DRMS and SRD datasets. 

## Requirements
Please install the following:
1. Tensorflow 1.10.1
2. skimage
3. opencv3

## Inference
The model expects 8bit RGB images of 256x256 pixels as inputs. In the evaluate.py file change the following:
1. L33: Point to the place where you download the pretrained model.
2. L35: Point to the path of the image to be tested

For ease of testing we already provide a sample image and the corresponding line in the script filled in.

The script can then be run from the command line as follows:
```
python evaluate.py
```

## Citation
Please cite the paper if it is useful in your research:
```
@article{Baslamisli\_Das\_2021,
 author = {A. ~S. Baslamisli and P. Das and H. ~A. Le and S. Karaoglu and T. Gevers},
 title = {ShadingNet: Image Intrinsics by Fine-Grained Shading Decomposition},
 journal = {International Journal of Computer Vision},
 volume = {129},
 number = {8},
 pages = {2445-2473},
 doi = {https://doi.org/10.1007/s11263-021-01477-5},
 year = 2021
}
```
