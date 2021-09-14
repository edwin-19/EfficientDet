import argparse
import preprocess

from dataset import FruitDatasetAdaptor, EffcientDetDataModule
from model import EffcientDetModel
import pytorch_lightning as pl

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
    model = EffcientDetModel()
    
    # Create trainer to train model
    trainer = pl.Trainer(
        gpus=[0], max_epochs=args.epoch, num_sanity_val_steps=1
    )
    trainer.fit(model, effdet_dm)