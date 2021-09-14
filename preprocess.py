import glob
import pandas as pd
import os
from xml.etree import ElementTree as ET
from typing import List
from sklearn.model_selection import train_test_split

def get_annots(annot_path: str, image_dir: str) -> pd.DataFrame:
    """
    Converts string of images 
    """
    annotations = glob.glob(os.path.join(annot_path, '*.xml'))
    return convert_annots(annotations, image_dir)

def convert_annots(annotations: List[str], image_dir: str) -> pd.DataFrame:
    """
    Function to convert xml of annotations to a dataframe for easier processing
    Args:
        annots: a list of string file paths
        image_dir: path to the image folder
        
    Returns: a daframe containing xy, height, width, and label class
    """
    data = {
        'image_path': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'labels': []
    }
    
    for annot in annotations:
        tree = ET.parse(annot)
        root = tree.getroot()
        file_name = root.find('filename').text
        image_path = glob.glob(os.path.join(image_dir, file_name))[0]
        
        for obj in root.findall('object'):
            data['image_path'].append(image_path)
            data['labels'].append(obj.find('name').text)
            for bb in obj.find('bndbox'):
                data[bb.tag].append(int(bb.text))
                
                
    return train_test_split(pd.DataFrame(data), test_size=0.2, random_state=2021)