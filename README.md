# CCVPE: Convolutional Cross-View Pose Estimation

This work is an extention of "Visual Cross-View Metric Localization with Dense Uncertainty Estimates, ECCV2022"

![](figures/overview.png)

<img src="figures/VIGOR_HFOV108.gif" width="300" height="450"/> <img src="figures/VIGOR_HFOV180.gif" width="300" height="450"/>
<img src="figures/VIGOR_HFOV360.gif" width="300" height="450"/>



### Abstract
We propose a novel end-to-end method for cross-view pose estimation. Given a ground-level query image and an aerial image that covers the query's local neighborhood, the 3 Degrees-of-Freedom camera pose of the query is estimated by matching its image descriptor to descriptors of local regions within the aerial image. The orientation-aware descriptors are obtained by using a translational equivariant convolutional ground image encoder and contrastive learning. The Localization Decoder produces a dense probability distribution in a coarse-to-fine manner with a novel Localization Matching Upsampling module. A smaller Orientation Decoder produces a vector field to condition the orientation estimate on the localization. Our method is validated on the VIGOR and KITTI datasets, where it surpasses the state-of-the-art baseline by 72% and 36% in median localization error for comparable orientation estimation accuracy. The predicted probability distribution can represent localization ambiguity, and enables rejecting possible erroneous predictions.
Without re-training, the model can infer on ground images with different field of views and utilize orientation priors if available. On the Oxford RobotCar dataset, our method can reliably estimate the ego-vehicle's pose over time, achieving a median localization error under 1 meter and a median orientation error of around 1 degree at 14 FPS.

### Datasets
VIGOR dataset can be found at https://github.com/Jeff-Zilence/VIGOR.
We use the revised ground truth from https://github.com/tudelft-iv/SliceMatch <br />
KITTI dataset can be found at https://github.com/shiyujiao/HighlyAccurate <br />
For Oxford RobotCar, the aerial image is provided by https://github.com/tudelft-iv/CrossViewMetricLocalization, the ground images are from https://robotcar-dataset.robots.ox.ac.uk/datasets/

### Models
Our trained models are avaiable at: https://surfdrive.surf.nl/files/index.php/s/cbyPn7NQoOOzlqp

### Training and testing
Training on VIGOR dataset: <br />
samearea split: `python train_VIGOR.py --area samearea` <br />
crossarea split: `python train_VIGOR.py --area crossarea` <br />

Test on VIGOR dataset: <br />
samearea split: `python train_VIGOR.py --area samearea --training False` <br />
crossarea split: `python train_VIGOR.py --area crossarea --training False` <br />


### Citations
```
@article{xia2023convolutional,
  title={Convolutional Cross-View Pose Estimation},
  author={Xia, Zimin and Booij, Olaf and Kooij, Julian FP},
  journal={arXiv preprint arXiv:2303.05915},
  year={2023}
}
```
