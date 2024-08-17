import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os

class CustomDataset(Dataset):

    def __init__(self, root, annotation, transform=None, classes=["__background__"]):
        self.root = root
        self.coco = COCO(annotation)
        self.classes = classes
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        coco_annotation = coco.loadAnns(ann_ids)
        coco_categories = coco.cats
        
        img_info = coco.loadImgs(img_id)[0]

        # replace backslashes in case annotations are made on Windows
        img_path = img_info["file_name"].replace("\\", "/")
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        num_objs = len(coco_annotation)
        boxes = []
        labels = []
        area = []

        # segmentation masks are expected to be of shape [N, H, W]
        masks = np.zeros((num_objs, img_info["height"], img_info["width"]), dtype=np.uint8)

        for i in range(num_objs):
            # boxes
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]

            boxes.append([xmin, ymin, xmax, ymax])

            # labels
            category_id = coco_categories[coco_annotation[i]["category_id"]]["name"]
            label = None
            if isinstance(self.classes, list):
                label = self.classes.index(category_id)
            elif isinstance(self.classes, dict):
                label = self.classes[category_id]

            if label is not None:
                labels.append(label)

            # masks
            masks[i,:,:] = coco.annToMask(coco_annotation[i])

            # area
            area.append(coco_annotation[i]["area"])
            
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = tv_tensors.Mask(torch.as_tensor(masks, dtype=torch.uint8))
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros(labels, dtype=torch.int64)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.ids)

