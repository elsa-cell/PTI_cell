from tqdm import tqdm
import logging
import os
import random
import json
from detectron2.structures import BoxMode




"""
This file contains functions to parse YOLO-format annotations into dicts in "Detectron2 format". FROM get_dicts in https://github.com/AugP-creatis/AdelaiDet-Z/blob/master/tools/train_net.py
"""




logger = logging.getLogger(__name__)

__all__ = ["get_dicts"]




# get_dicts and register_datasets from Nathan Hutin https://gitlab.in2p3.fr/nathan.hutin/detectron2/-/blob/main/train_cross_validation.py
# inspired from official Detectron2 tutorial notebook https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# /!\ ATTENTION !! LE SYSTEME DE FICHIER A ETE CHANGE, LA DOCUMENTATION AUSSI A ETE MDIFIEE POUR COLLER AUX ARGUMENTS REELS DE LA FONCTION

def get_dicts(dir, mode, idx_cross_val, classes):
    """
    Read the annotations for the dataset in YOLO format and create a list of dictionaries containing information for each
    image.

    Args:
        dir (str) : Path to the directory containing the data, structured as follows:
            |___ Cross-val
                    |___ Xval0
                    |       |__ images
                    |       |__ labels
                    |___ Xval1
                    |       |__ images
                    |       |__ labels
                    |___ Xval2
                    |       |__ images
                    |       |__ labels
                    |___ Xval3
                    |       |__ images
                    |       |__ labels
                    |___ Xval4
                            |__ images
                            |__ labels
        mode (str): 'train' : for training data, 'val' : for evaluation data, '' : for testing data
        idx_cross_val (int): 0 to 4, 
            if 0 => train contains folds 2,3,4 ; val contains fold 1 ; test contains fold 0
            if 1 => train contains folds 3,4,0 ; val contains fold 2 ; test contains fold 1
            and so on...
        classes (list): classes correspondance to the labels
        

    Returns:
        list[dict]: A list of dictionaries containing information for each image. Each dictionary has the following keys:
            - file_name: The path to the image file.
            - image_id: The unique identifier for the image.
            - height: The height of the image in pixels.
            - width: The width of the image in pixels.
            - annotations: A list of dictionaries, one for each object in the image, containing the following keys:
                - bbox: A list of four integers [x0, y0, w, h] representing the bounding box of the object in the image,
                        where (x0, y0) is the top-left corner and (w, h) are the width and height of the bounding box,
                        respectively.
                - bbox_mode: A constant from the `BoxMode` class indicating the format of the bounding box coordinates
                             (e.g., `BoxMode.XYWH_ABS` for absolute coordinates in the format [x0, y0, w, h]).
                - category_id: The integer ID of the object's class.
    """
    random.seed(0)
    if mode == 'train':
        cross_val_dict = {0:[2,3,4], 1:[0,3,4], 2:[0,1,4], 3:[0,1,2], 4:[1,2,3]}
        folds_list = cross_val_dict[idx_cross_val]

    elif mode == 'val' :
        cross_val_dict = {0:[1], 1:[2], 2:[3], 3:[4], 4:[0]}
        folds_list = cross_val_dict[idx_cross_val]
    
    else:
        cross_val_dict = {0:[0], 1:[1], 2:[2], 3:[3], 4:[4]}
        folds_list = cross_val_dict[idx_cross_val]

    dataset_dicts = []
    dict_instance_label = {value:num for num, value in enumerate(classes.values())}
    for fold in folds_list:
        img_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold), 'images')
        ann_dir = os.path.join(dir, 'Cross-val', 'Xval'+str(fold),'labels')
        
        if not os.path.exists(img_dir):
            print(f"The path {img_dir} does not exist.")
        if not os.path.exists(ann_dir):
            print(f"The path {ann_dir} does not exist.")

        for idx, file in tqdm(enumerate(os.listdir(ann_dir)), desc=f'cross validation {fold}, mode {mode}'):
            # annotations should be provided in yolo format
            if mode !='train' and 'Augmented' in file:
                continue

            record = {}
            dico = json.load(open(os.path.join(ann_dir, file)))

            record["file_name"] = os.path.join(img_dir, dico['info']['filename'])
            record["image_id"] = dico['info']['image_id']
            record["height"] = dico['info']['height']
            record["width"] = dico['info']['width']

            objs = []
            for instance in dico['annotation']:
                if 'Trash' in classes.keys() and instance['category_id'] in classes['Trash']:
                    instance['category_id'] = 1

                if instance['category_id'] in classes.values() or ('trash' in classes.keys() and instance['category_id'] in classes['trash']):

                    obj = {
                        "bbox": instance['bbox'],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": dict_instance_label[instance['category_id']],
                        'segmentation' : instance['segmentation']
                    }

                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)


    return dataset_dicts