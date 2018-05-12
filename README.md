# PyTorch implementation of the [OpenPose](https://arxiv.org/abs/1611.08050)

The OpenPose is one of the most popular keypoint estimator, which uses two branches of feature map (is trained and enhanced via multiple stages) to estimate (via a [postprocess procedure](https://github.com/ruiminshen/pyopenpose)) the position of keypoints (via Gaussian heatmap) and the relationship between keypoints (called part affinity fields), respectively.
This project adopts [PyTorch](http://pytorch.org/) as the developing framework to increase productivity, and utilize [ONNX](https://github.com/onnx/onnx) to convert models into [Caffe 2](https://caffe2.ai/) to benefit engineering deployment.
If you are benefited from this project, a donation will be appreciated (via [PayPal](https://www.paypal.me/minimumshen), [微信支付](donate_mm.jpg) or [支付宝](donate_alipay.jpg)).

## Designs

- Flexible configuration design.
Program settings are configurable and can be modified (via **configure file overlaping** (-c/--config option) or **command editing** (-m/--modify option)) using command line argument.

- Monitoring via [TensorBoard](https://github.com/tensorflow/tensorboard).
Such as the loss values and the debugging images (such as IoU heatmap, ground truth and predict bounding boxes).

- Parallel model training design.
Different models are saved into different directories so that can be trained simultaneously.

- Time-based output design.
Running information (such as the model, the summaries (produced by TensorBoard), and the evaluation results) are saved periodically via a predefined time.

- Checkpoint management.
Several latest checkpoint files (.pth) are preserved in the model directory and the older ones are deleted.

- NaN debug.
When a NaN loss is detected, the running environment (data batch) and the model will be exported to analyze the reason.

- Unified data cache design.
Various dataset are converted into a unified data cache via a programmable (a series of Python lambda expressions, which means some points can be flexibly generated) configuration.
Some plugins are already implemented. Such as [MS COCO](http://cocodataset.org/).

- Arbitrarily replaceable model plugin design.
The deep neural network (both the feature extraction network and the stage networks) can be easily replaced via configuration settings.
Multiple models are already provided. Such as the oringal VGG like network, [Inception v4](https://arxiv.org/abs/1602.07261), [MobileNet v2](https://arxiv.org/abs/1801.04381) and [U-Net](https://arxiv.org/abs/1505.04597).

- Extendable data preprocess plugin design.
The original images (in different sizes) and labels are processed via a sequence of operations to form a training batch (images with the same size, and bounding boxes list are padded).
Multiple preprocess plugins are already implemented. Such as
augmentation operators to process images and labels (such as random rotate and random flip) simultaneously,
operators to resize both images and labels into a fixed size in a batch (such as random crop),
and operators to augment images without labels (such as random blur, random saturation and random brightness).

## Quick Start

This project uses [Python 3](https://www.python.org/). To install the dependent libraries, make sure the [pyopenpose](https://github.com/ruiminshen/pyopenpose) is installed, and type the following command in a terminal.

```
sudo pip3 install -r requirements.txt
```

`quick_start.sh` contains the examples to perform detection and evaluation. Run this script.
The COCO dataset is downloaded ([aria2](https://aria2.github.io/) is required) and cached, and the original pose model (18 parts and 19 limbs) is converted into PyTorch's format.
If a webcam is present, the keypoint estimation demo will be shown.
Finally, the training program is started.
