import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from maskrcnn.dataset import CustomDataset
from maskrcnn.utils.transforms import ToTensor, Compose
from maskrcnn.models.maskrcnn_resnet50_fpn import create_maskrcnn_resnet50_fpn

class MaskRCNN():

    def __init__(self, 
                 model_path,
                 backbone="resnet50-fpn",
                 epochs=10,
                 batch_size=4, 
                 learning_rate=0.001, 
                 classes=["__background__"],
                 annotations_path="annotations/train.json",
                 optimizer="sgd",
                 scheduler="step_lr",
                 output_state_dict_name=None,
                 clip_gradients=False,
                 scheduler_step_size=7,
                 scheduler_patience=2,
                 sgd_momentum=0.9,
                 sgd_weight_decay=0.0001
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classes = classes
        self.annotations_path = annotations_path
        self.data_loader = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.__load_model(model_path=model_path)

        # if True, this will clip gradient values with clip_grad_norm_ during training
        self.clip_gradients = clip_gradients

        # optimizer
        self.optimizer = None
        self.target_optimizer = optimizer

        # scheduler
        self.scheduler = None
        self.target_scheduler = scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_patience = scheduler_patience

        # SGD parameters (if preferred optimizer is SGD)
        self.sgd_momentum = sgd_momentum
        self.sgd_weight_decay = sgd_weight_decay
        
        # filename for the output state dictionary
        if output_state_dict_name is None:
            self.output_state_dict_name = f"mask_rcnn_{backbone}_{batch_size}x{epochs}.pth"
        else:
            self.output_state_dict_name = output_state_dict_name


    def __load_model(self, model_path):
        self.model = create_maskrcnn_resnet50_fpn(num_classes=len(self.classes), 
                                                  path_to_weights=model_path)

    
    def __load_optimizer(self, target_optimizer):
        if self.model is None:
            print("Trying to load optimizer without an initialized model")
            raise SystemExit(1)

        params = [p for p in self.model.parameters() if p.requires_grad]

        if target_optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif target_optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.learning_rate)
        
        # use SGD for the optimizer by default
        return torch.optim.SGD(params,
                               lr=self.learning_rate,
                               momentum=self.sgd_momentum,
                               weight_decay=self.sgd_weight_decay)


    def __load_scheduler(self, target_scheduler):
        if self.optimizer is None:
            print("Trying to load scheduler without an initialized optimizer")
            raise SystemExit(1)

        if target_scheduler == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(self.optimizer, "min", patience=self.scheduler_patience)

        # use StepLR by default
        return StepLR(self.optimizer, step_size=self.scheduler_step_size, gamma=0.1)


    def get_transform(self):
        transforms = []

        # add additional transforms here
        transforms.append(ToTensor())

        return Compose(transforms)


    def __train_epoch(self, curr_epoch):
        print("Training started")
        if self.model is None:
            print("Model not initialized")
            raise SystemExit(1)

        if self.optimizer is None:
            print("Optimizer is not defined")
            raise SystemExit(1)

        if self.data_loader is None:
            print("Data Loader is not defined")
            raise SystemExit(1)

        self.model.train()

        batch_idx = 0
        epoch_loss = 0

        for images, targets in self.data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            if self.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            print(f"Epoch {curr_epoch}/{self.epochs}, Running Loss: {losses.item()}")

            batch_idx += 1
            epoch_loss += losses.item()

        return epoch_loss


    def train(self):
        if self.model is None:
            print("Model not initialized")
            raise SystemExit(1)
        
        dataset = CustomDataset(
                root=".",
                annotation=self.annotations_path,
                transform=self.get_transform(),
                classes=self.classes)

        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda x: tuple(zip(*x)))

        self.optimizer = self.__load_optimizer(self.target_optimizer)
        self.scheduler = self.__load_scheduler(self.target_scheduler)

        if self.scheduler is None:
            print("Scheduler not initialized")
            raise SystemExit(1)

        for epoch_idx in range(self.epochs):
            epoch = epoch_idx + 1
            epoch_loss = self.__train_epoch(epoch)
            avg_loss = epoch_loss / len(self.data_loader)

            print(f"[Epoch {epoch} Results] Avg. Loss: {avg_loss}")

            if self.target_scheduler == "step_lr":
                self.scheduler.step()
            elif self.target_scheduler == "reduce_lr_on_plateau":
                # TODO: support for a criterion?
                self.scheduler.step(avg_loss)

        torch.save(self.model.state_dict(), self.output_state_dict_name)
