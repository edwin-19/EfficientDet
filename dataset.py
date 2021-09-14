import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
import cv2
import numpy as np
import torch
import utils
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

#### Transformation functions ####
def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc', min_area=0, min_visibility=0, label_fields=["labels"]
        )
    )
    
def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        )
    )


#### Dataset Features ####
class FruitDatasetAdaptor(object):
    def __init__(self, annot_df):
        self.annot_df = annot_df
        self.labels = {
            'orange': 0,
            'banana': 1,
            'apple': 2
        }
    
    def __len__(self):
        return self.annot_df.shape[0]
    
    def get_image_and_labels_by_idx(self, index):
        annot = self.annot_df.iloc[index]
        image_path = annot['image_path']
        
        # Used opencv because requires bgr as an input format
        # For albumentations
        img = cv2.imread(image_path)
        
        selected_annotations = self.annot_df[self.annot_df['image_path'] == image_path]
        pascal_bboxes = selected_annotations[
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = selected_annotations['labels'].values
        class_labels = [self.labels[class_label] for class_label in class_labels]
        return img, pascal_bboxes, class_labels, index
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        utils.show_image(image, bboxes.tolist(), class_labels)
        
        
class EfficientDetDataset(Dataset):
    def __init__(
        self,
        dataset_adaptor, 
        transforms=get_valid_transforms()
    ):
        super(EfficientDetDataset, self).__init__()
        self.ds = dataset_adaptor
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        (
            image, pascal_bboxes, 
            class_labels, image_id
        ) = self.ds.get_image_and_labels_by_idx(index)
        
        sample = {
            'image': image.astype(np.float32),
            'bboxes': pascal_bboxes,
            'labels': class_labels
        }
        
        sample = self.transforms(**sample)
        sample['bboxes'] = np.array(sample['bboxes'])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]
        
        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx
        
        target = {
            "bboxes": torch.as_tensor(pascal_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0])
        }
        
        return image, target, image_id

class EffcientDetDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dataset_adaptor, validation_dataset_adaptor,
        train_transforms=get_train_transforms(target_img_size=512), valid_transforms=get_valid_transforms(target_img_size=512),
        num_workers=4, batch_size=8
    ):
        super(EffcientDetDataModule, self).__init__()
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        ) 
        
    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        random_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=random_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        return train_loader
    
    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )
        
    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        seq_sampler = SequentialSampler(valid_dataset)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=seq_sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        
        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            'bbox': boxes,
            'cls': labels,
            'img_size': img_size,
            'img_scale': img_scale
        }
        
        return images, annotations, targets, image_ids