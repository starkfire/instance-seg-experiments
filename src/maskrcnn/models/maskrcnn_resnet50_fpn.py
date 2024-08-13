import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def create_maskrcnn_resnet50_fpn(num_classes, path_to_weights=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    
    # replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes=num_classes)
    
    if path_to_weights is not None:
        ckpt = torch.load(path_to_weights, map_location=device)
        model.load_state_dict(ckpt)

    model.to(device)

    return model
