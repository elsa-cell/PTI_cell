#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

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


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
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
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.default_cell_stack:
        cfg = get_stack_cell_config(cfg)
    if args.classes_dict:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(eval(args.classes_dict))
    
    cfg.MODEL.RESNETS.DEPTH = 18                                 # Configuration of the depth of the resnet network, default to 50, for lighter models, use 18
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.WEIGHTS = ""
    '''
    cfg.MODEL.PIXEL_MEAN = [218.96615195, 205.58776696, 199.45428186]
    cfg.MODEL.PIXEL_STD = [20.1397195,  19.81115748, 21.38672478]
    '''

    # SOLVER
    cfg.SOLVER.IMS_PER_BATCH = 24          # Attention à la taille de la mémoire dont dispose la GPU, doit aussi être un multiple du nombre de GPU
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 20
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.REFERENCE_WORLD_SIZE = args.num_gpus
    cfg.MODEL.USE_AMP = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Seuil de détection : Seulement les détections avec score > 0.5
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD
    cfg.TEST.COMPUTE_LOSSES = False

    cfg.merge_from_list(args.opts)
    
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = {18:64, 32:64, 50:256, 101:256, 152:256}[cfg.MODEL.RESNETS.DEPTH]
    if (cfg.SOLVER.IMS_PER_BATCH % args.num_gpus != 0):    # Pour être sûr d'être divisible par le nombre de GPU
        cfg.SOLVER.IMS_PER_BATCH = (cfg.SOLVER.IMS_PER_BATCH // args.num_gpus) * args.num_gpus
        
    cfg.freeze()
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
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser()

    args.add_argument('--default-cell-stack', default=True)
    args.add_argument('--data-dir', default='/projects/INSA-Image/B01/Data/')
    args.add_argument('--classes-dict',type=str,default="{'Intact_Sharp':0, 'Broken_Sharp':2}")
    #Classes are like "{'Intact_Sharp':0,'Intact_Blurry':1,'Broken_Sharp':2,'Broken_Blurry':3}"
    args.add_argument('--cross-val', default=4)

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
