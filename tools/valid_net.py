#!/usr/bin/env python3
"""
Custom validation script. Runs on every weight file placed into the 
"""

import logging
import os
from collections import OrderedDict
import torch
import glob

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, get_stack_cell_config
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
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

class CustomValidationTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. This class is helpfull to define some evaluators
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    setup_logger(name=__name__)
    
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(args.weight_dir_path, "config.yaml"))

    cfg.MODEL.WEIGHTS = ""

    #cfg.SOLVER.IMS_PER_BATCH = 4          # Attention à la taille de la mémoire dont dispose la GPU, doit aussi être un multiple du nombre de GPU
    #cfg.SOLVER.REFERENCE_WORLD_SIZE = args.num_gpus
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Seuil de détection : Seulement les détections avec score > 0.5
    #cfg.TEST.COMPUTE_LOSSES = False

    if hasattr(args, "MODEL.ROI_HEADS.SCORE_THRESH_TEST"):
        cfg.merge_from_list([("MODEL.ROI_HEADS.SCORE_THRESH_TEST", args.MODEL_ROI_HEADS_SCORE_THRESH_TEST)])
    if hasattr(args, "SOLVER.IMS_PER_BATCH"):
        cfg.merge_from_list([("SOLVER.IMS_PER_BATCH", args.SOLVER.IMS_PER_BATCH)])
    if hasattr(args, "TEST.COMPUTE_LOSSES"):
        cfg.merge_from_list([("TEST.COMPUTE_LOSSES", args.TEST.COMPUTE_LOSSES)])
    #cfg.merge_from_list([arg for arg in args if arg.startswith("SOLVER.IMS_PER_BATCH")])

    if args.num_gpus != 0:
        if (cfg.SOLVER.IMS_PER_BATCH % args.num_gpus != 0):    # Pour être sûr d'être divisible par le nombre de GPU
            cfg.SOLVER.IMS_PER_BATCH = (cfg.SOLVER.IMS_PER_BATCH // args.num_gpus) * args.num_gpus

    default_setup(cfg, args)
    
    return cfg


def main(args):
    cfg = setup(args)

    classes = eval(args.classes_dict)

    # If the dataset is custom, adapt the registering function
    if cfg.DATALOADER.IS_STACK:
        DatasetCatalog.register('train', lambda: get_dicts(args.data_dir, 'train', args.cross_val, classes))
        DatasetCatalog.register('val', lambda: get_dicts(args.data_dir, 'val', args.cross_val, classes))
        # Set the metadata for the dataset.
        MetadataCatalog.get('train').set(thing_classes=list(classes.keys()))
        MetadataCatalog.get('val').set(thing_classes=list(classes.keys()), evaluator_type="coco")
    
    model = CustomValidationTrainer.build_model(cfg)

    AP_75 = 0
    path_to_recommanded_weights = ""

    all_weights_paths = sorted(glob.glob(os.path.join(args.weight_dir_path, 'model_*.pth')))
    nb_weight_files = len(all_weights_paths)
    metrics_dic = {}
    for (i, weights_path) in zip(range(len(all_weights_paths)), all_weights_paths):
        logger.info("Iteration {}/{}".format(i+1, nb_weight_files))
        logger.info("Using weights stored in {}".format(weights_path))
        cfg.MODEL.WEIGHTS = weights_path
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        metrics = CustomValidationTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            metrics.update(CustomValidationTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, metrics)
            
        #logger.info("Keys in metrics: {}".format(metrics.keys()))
        #logger.info("Keys in metrics['segm']: {}".format(metrics['segm'].keys()))
        # TODO, faire en sorte d'évaluer en fonction d'une métrique passée en paramètre
        # Metric can be ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Intact_Sharp', 'AP-Broken_Sharp']
        #curr_AP_75 = metrics['segm']['AP75']
        curr_AP_75 = metrics[args.valid_type][args.valid_category]
        if AP_75 < curr_AP_75:
            AP_75 = curr_AP_75
            path_to_recommanded_weights = weights_path

        metrics_dic.update({i:metrics})

    json_file_name = os.path.join(cfg.OUTPUT_DIR, 'validation_metrics.json')
    with open(json_file_name, 'w') as json_file:
        json.dump(metrics_dic, json_file, indent=4)

    logger.info("Recommended weights are stored in {}".format(path_to_recommanded_weights))
    
    return metrics, path_to_recommanded_weights


if __name__ == "__main__":
    args = default_argument_parser()

    args.add_argument('--weight-dir-path', default="/tmp/TEST/outputs/3D_50_layers/", help="path to the directory containing the full config file created when training, as well as different versions of the model weights")
    args.add_argument('--data-dir', default='/projects/INSA-Image/B01/Data/')
    args.add_argument('--classes-dict',type=str,default="{'Intact_Sharp':0, 'Broken_Sharp':2}")
    #Classes are like "{'Intact_Sharp':0,'Intact_Blurry':1,'Broken_Sharp':2,'Broken_Blurry':3}"
    args.add_argument('--cross-val', default=4, help="will be set by default to the one used during training to avoid runing validation on test data. If wasn't saved when training, will be the one specified here.")
    args.add_argument('--valid-type', default='segm', help="can be set to 'segm' or to 'bbox' to have metrics based on the segmentations or the bounding boxes. Only valid for coco evauator")
    args.add_argument('--valid-category', default='AP75', help="can be set to 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Intact_Sharp', 'AP-Broken_Sharp'. Only valid for coco evauator")


    args = args.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
