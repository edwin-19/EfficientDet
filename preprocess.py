import glob
import pandas as pd
import os
from xml.etree import ElementTree as ET

def convert_annots(annotations_path: str) -> pd.DataFrame:
    """
    Function to convert xml of annotations to a dataframe for easier processing
    Args:
        annots: a list of string file paths
        image_dir: path to the image folder
        
    Returns: a daframe containing xy, height, width, and label class
    """
    
    annotations = glob.glob(os.path.join(annotations_path, '*.xml'))
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
        image_path = glob.glob(os.path.join(annotations_path, file_name))[0]
        
        for obj in root.findall('object'):
            data['image_path'].append(image_path)
            data['labels'].append(obj.find('name').text)
            for bb in obj.find('bndbox'):
                data[bb.tag].append(int(bb.text))
                
                
    return pd.DataFrame(data)