import argparse

import torch
import preprocess
import os

from dataset import FruitDatasetAdaptor, EffcientDetDataModule
from dataset import get_train_transforms, get_valid_transforms
from model import EffcientDetModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/archive/train_zip/train')
    parser.add_argument('--test_dir', default='data/archive/test_zip/test')
    parser.add_argument('--epoch', type=int, default=15)
    args = parser.parse_args()
    
    train_df = preprocess.convert_annots(args.train_dir)
    test_df = preprocess.convert_annots(args.test_dir)
    
    # Create dataset adaptor
    train_ds = FruitDatasetAdaptor(train_df)
    test_ds = FruitDatasetAdaptor(test_df)
    
    # Create data module
    effdet_dm = EffcientDetDataModule(
        train_ds, test_ds, num_workers=4, batch_size=2,
        train_transforms=get_train_transforms(target_img_size=512),
        valid_transforms=get_valid_transforms(target_img_size=512)
    )
    
    # Create model
    model = EffcientDetModel(
        task='train', model_architecture='tf_efficientdet_d1', learning_rate=2e-4
    )
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='valid_loss_epoch',
        min_delta=0.02,
        patience=5,
        verbose=True,
        mode='max'
    )

    # Create trainer class to train model
    trainer = pl.Trainer(
        gpus=[0], max_epochs=args.epoch, num_sanity_val_steps=1,
        progress_bar_refresh_rate=20,
        callbacks=[early_stopping]
    )
    trainer.fit(model, effdet_dm)
    
    # Save weights
    if not os.path.exists('weights'):
        os.makedirs('weights/')
        
    torch.save(model.state_dict(), 'weights/effdet_fruits_l.pth')
    # trainer.save_checkpoint('weights/effdet_l.ckpt')