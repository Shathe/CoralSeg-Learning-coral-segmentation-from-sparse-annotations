# CoralSeg: Learning coral segmentation from sparse annotations
## Requirements

- tensorflow
- imgaug
- cv2
- keras
- scipy
- matplotlib

## Citing CoralSeg

[Link to the paper](http://csms.haifa.ac.il/profiles/tTreibitz/webfiles/CoralSeg2019.pdf)

If you find our work useful in your research, please consider citing:
```
@inproceedings{alonso2019CoralSeg,
  title={CoralSeg: Learning Coral Segmentation from Sparse Annotations},
  author={Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and Murillo, Ana C},
  booktitle={Journal of Field Robotics},
  year={2019}
}
```

## Web project and Downloadable content (Weights, datasets...)
For downloading the trained weights and our datasets, please go to [our website project](https://sites.google.com/a/unizar.es/semanticseg/home/)

## Multi-Level Superpixel Augmentation
The github repository concerning the labeling augmentation/propagation method is in [here](https://github.com/Shathe/ML-Superpixels).

## Training CoralSeg from Scratch
This repository already contains the EilatMixx dataset in the Datasets folder.

```
python train.py
```
This scripts has different arguments for specifying the dataset, the model path, the input resolution and other configuration parameters, feel free to play with them.

The training scripts gives test metrics in every epoch, but if you want to test it onced is trained, just change the *train* parameter to *0*.

## Training CoralSeg from CoralNet weights

First of all, download [these weights](https://drive.google.com/file/d/1x5dktklVLVwkomZxYj7ncFCe-JL5bX0l/view?usp=sharing) and save them in the models folder.
And the just execute: 
```
python train_pretrained.py
```
In the script, the *checkpoint_path* parameter is set to models/deeplab_encoder that is where the trained weights should be.


