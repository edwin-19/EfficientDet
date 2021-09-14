import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
import numpy as np
import utils

#### Transformation functions ####
def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(target_img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc', min_area=0, min_visibility=0, label_fields=["labels"]
        )
    )
    
def get_valid_transforms(target_img_size=512):
    pass


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
        
        img = Image.open(image_path)
        
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