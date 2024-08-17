# Instance Segmentation Experiments

Template repository for setting up a ready-to-use Mask R-CNN training/inference pipeline in PyTorch.

## Backbones
* ResNet50
* ResNet18
* MobileNetV2

## Usage

A default training class can be defined like so:

```py
model = MaskRCNN()
```

### Using a different backbone

By default, this project uses ResNet50 for the default backbone. However, given its large parameter count, you may sometimes need to strike the balance between performance and accuracy. As a result, you may want to train using a different backbone.

You can use the following backbones:

* `resnet50-fpn` (ResNet50-FPN) (default)
* `resnet18-fpn` (ResNet18-FPN)

Specify your target backbone using the `backbone` parameter:

```py
# ResNet50-FPN
model = MaskRCNN(backbone="resnet50-fpn")

# ResNet18-FPN
model = MaskRCNN(backbone="resnet18-fpn")
```

**Note:** ResNet18 does not come with its own FPN. You may want to check `src/models/maskrcnn_resnet18_fpn.py` to see if the existing backbone implementation matches your needs.

### Passing your own finetuned weights

You can also use your own weights by passing a path to your `.pth` file.

```py
model = MaskRCNN("./weights.pth", backbone="resnet50-fpn")
```

Note that `backbone` needs to match the backbone that is used with the state dictionary.
