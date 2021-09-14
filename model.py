import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import timm
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet

def get_timm_model_listing(model_architecture_prefix='tf_efficientnetv2_*'):
    return timm.list_models(model_architecture_prefix)

def create_model(
    num_classes=3, image_size=512, architecture='tf_efficientnetv2_l'
):
    """
    Function to create a effnet model
    
    Args:
        num_classes: num of classes to detect
        image_size: size of the image must be divisable by 128
        architecture: the model architecture specified for use
    """
    efficientdet_model_param_dict[architecture] = {
        'name': architecture,
        'backbone_name': architecture,
        'backbone_args': {
            'drop_path_rate':0.2
        },
        'num_classes': num_classes,
        'url': ''
    }
    
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config, num_outputs=config.num_classes
    )
    
    return DetBenchTrain(net, config)

class EffcientDetModel(pl.LightningModule):
    def __init__(
        self, num_classes=3, img_size=512,
        prediction_confidence_threshold=0.2,learning_rate=0.0002,
        model_architecture='tf_efficientnetv2_l',
    ):
        super(EffcientDetModel, self).__init__()
        
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
    
    @auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr
        )
        
    def training_step(self, batch, batch_idx):
        images, annotations, _, _ = batch
        losses = self.model(images, annotations)
        
        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(
            "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True
        )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        
        detections = outputs["detections"]
        
        batch_predictions = {
            'predictions': detections,
            'targets': targets,
            'image_ids': image_ids
        }
        
        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }
        
        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        
        return {
            'loss': outputs['loss'],
            'batch_predictions': batch_predictions
        }