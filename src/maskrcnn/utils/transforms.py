from torchvision.transforms import functional as F
import random

# custom implementation of T.Compose() which takes
# image and target
class Compose(object):
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# custom implementation of T.ToTensor() which takes
# image and target
class ToTensor(object):

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# custom implementation of T.RandomHorizontalFlip() which
# takes image and target
class RandomHorizontalFlip(object):

    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        
        return image, target

