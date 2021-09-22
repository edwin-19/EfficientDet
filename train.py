import argparse
import preprocess
import os

from dataset import FruitDatasetAdaptor, EffcientDetDataModule
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
        train_ds, test_ds, num_workers=4, batch_size=2
    )
    
    # Create model
    model = EffcientDetModel(task='train')    
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='valid_loss_epoch',
        min_delta=0.02,
        patience=10,
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
    if os.path.exists('weights'):
        os.makedirs('weights/')
        
    trainer.save_checkpoint('weights/effdet_l.ckpt')