#!/usr/bin/env python3
"""
Custom validation script. Runs on every weight file placed into the 
"""

import logging
import os
from collections import OrderedDict
import torch
import glob
import json

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, get_stack_cell_config
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from tools.train_net import Trainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import get_dicts
from detectron2.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    setup_logger(name=__name__)
    
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(args.weight_dir_path, "config.yaml"))
    if cfg.is_frozen():
        cfg.defrost()

    cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

    if len(args.opts):
        opts_dict = {}
        # Gets the keys and values
        for i in range(0, len(args.opts), 2):
            key = args.opts[i]
            value = args.opts[i+1]
            opts_dict[key] = value
        if opts_dict.get("MODEL.ROI_HEADS.SCORE_THRESH_TEST"):
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opts_dict['MODEL.ROI_HEADS.SCORE_THRESH_TEST']
        if opts_dict.get("SOLVER.IMS_PER_BATCH"):
            cfg.SOLVER.IMS_PER_BATCH = opts_dict["SOLVER.IMS_PER_BATCH"]
        if opts_dict.get("TEST.COMPUTE_LOSSES"):
            cfg.TEST.COMPUTE_LOSSES = opts_dict["TEST.COMPUTE_LOSSES"]
        if opts_dict.get("OUTPUT_DIR"):
            cfg.OUTPUT_DIR = opts_dict["OUTPUT_DIR"]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    

    default_setup(cfg, args)
    
    return cfg


def main(args):
    cfg = setup(args)

    classes = eval(args.classes_dict)

    idx_cross_val = 0
    if args.eval_only:
        # Pour que le dataset de test soit registered à la place du dataset de validation. Il aura le nom 'val'
        correspondance_idx = {0:4, 1:0, 2:1, 3:2, 4:3}
        idx_cross_val = correspondance_idx[args.cross_val]
    else:
        idx_cross_val = args.cross_val

    # If the dataset is custom, adapt the registering function
    if cfg.DATALOADER.IS_STACK:
        DatasetCatalog.register('train', lambda: get_dicts(args.data_dir, 'train', idx_cross_val, classes))
        DatasetCatalog.register('val', lambda: get_dicts(args.data_dir, 'val', idx_cross_val, classes))
        # Set the metadata for the dataset.
        MetadataCatalog.get('train').set(thing_classes=list(classes.keys()))
        MetadataCatalog.get('val').set(thing_classes=list(classes.keys()), evaluator_type="coco")
    
    model = Trainer.build_model(cfg)

    best_AP_75 = 0
    path_to_recommanded_weights = ""

    all_weights_paths = ""
    if args.eval_only:
        all_weights_paths = [os.path.join(args.weight_dir_path, args.weight_file)]
    else:
        all_weights_paths = sorted(glob.glob(os.path.join(args.weight_dir_path, 'model_*.pth')))
    nb_weight_files = len(all_weights_paths)
    metrics_dic = {}
    for (i, weights_path) in zip(range(len(all_weights_paths)), all_weights_paths):
        logger.info("Iteration {}/{}".format(i+1, nb_weight_files))
        cfg.MODEL.WEIGHTS = weights_path
        logger.info("Using weights stored in {}".format(weights_path))
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
        metrics = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            metrics.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, metrics)
            
        #logger.info("Keys in metrics: {}".format(metrics.keys()))
        #logger.info("Keys in metrics['segm']: {}".format(metrics['segm'].keys()))
        # TODO, faire en sorte d'évaluer en fonction d'une métrique passée en paramètre
        # Metric can be ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Intact_Sharp', 'AP-Broken_Sharp']
        #curr_AP_75 = metrics['segm']['AP75']
        curr_AP_75 = metrics[args.valid_type][args.valid_category]
        if best_AP_75 < curr_AP_75:
            best_AP_75 = curr_AP_75
            path_to_recommanded_weights = weights_path

        metrics_dic.update({i:metrics})

    if args.eval_only:
        json_file_name = os.path.join(cfg.OUTPUT_DIR, 'test_metrics.json')
    else:
        json_file_name = os.path.join(cfg.OUTPUT_DIR, 'validation_metrics.json')
    with open(json_file_name, 'w') as json_file:
        json.dump(metrics_dic, json_file, indent=4)

    logger.info("Recommended weights are stored in {}".format(path_to_recommanded_weights))
    
    return metrics, path_to_recommanded_weights

    

    
if __name__ == "__main__":
    args = default_argument_parser()

    args.add_argument('--weight-dir-path',type=str, default='/tmp/TEST/outputs/3D_50_layers/', help="path to the directory containing the full config file created when training, as well as different versions of the model weights")
    args.add_argument('--data-dir',type=str, default='/projects/INSA-Image/B01/Data/')
    args.add_argument('--classes-dict',type=str,default="{'Intact_Sharp':0, 'Broken_Sharp':2}")
    #Classes are like "{'Intact_Sharp':0,'Intact_Blurry':1,'Broken_Sharp':2,'Broken_Blurry':3}"
    args.add_argument('--cross-val',type=int, default=4, help="will be set by default to the one used during training to avoid runing validation on test data. If wasn't saved when training, will be the one specified here.")
    args.add_argument('--valid-type',type=str, default='segm', help="can be set to 'segm' or to 'bbox' to have metrics based on the segmentations or the bounding boxes. Only valid for coco evauator")
    args.add_argument('--valid-category',type=str, default='AP75', help="can be set to 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Intact_Sharp', 'AP-Broken_Sharp'. Only valid for coco evauator")
    args.add_argument('--weight-file',type=str, default='', help="has to be set if test-only")


    args = args.parse_args()

    setup_logger(name=__name__)
    logger.info("Command Line Args: {}".format(args))
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
