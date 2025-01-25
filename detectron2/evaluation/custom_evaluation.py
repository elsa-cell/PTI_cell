import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.utils.logger import log_first_n
from .coco_evaluation import COCOEvaluator


class CustomEvaluator(COCOEvaluator):
    """
    Results:
    {
    "bbox": {"AP": 0.5, "AP50": 0.6, ...},   # COCO results
    "segm": {"AP": 0.4, "AP50": 0.55, ...},  # COCO results
    "dice_scores": [0.8, 0.7, 0.9, ...]      # Dice
    }
    """

    def __init__(self, dataset_name, cfg, distributed=True, output_dir=None):
        # Initialisation de l'évaluateur COCO standard
        super().__init__(dataset_name, cfg, distributed, output_dir)

    def reset(self):
        super().reset(self)

    def process(self, inputs, outputs):
        """
        Process function to compute the Dice score in addition to the standard COCO metrics.
        """
        super().process(self, inputs, outputs)

        self._gt_masks = []
        for input in inputs:
            instances = input["instances"]
            if instances.has("gt_masks"):
                self._gt_masks.append(instances.gt_masks)

        self._pred_masks = []
        for output in outputs:
            instances = output["instances"]
            self._pred_masks.append(instances.pred_masks)
            
            
    def evaluate(self):
        """
        Coco evaluation and Dice scores
        """
        results = super().evaluate(self)


        # Calcul du score de Dice pour chaque paire de masque de vérité terrain et de prédiction
        dice_scores = []
        for gt, pred in zip(self._gt_masks, self._pred_masks):
            for gt_mask in gt:
                best_dice = 0.0
                predicted = False
                for pred_mask in pred:
                    # Calculer l'IoU entre le masque de ground truth et le masque prédit
                    intersection = torch.sum(gt_mask & pred_mask).float()
                    union = torch.sum(gt_mask | pred_mask).float()

                    if intersection/union > 0.75:
                        predicted = True
                        dice = 2. * intersection / (union + intersection)
                        best_dice = max(best_dice, dice)  # Prendre le meilleur score de Dice pour chaque masque de ground truth
                if predicted:
                    dice_scores.append(best_dice)


        results["dice_scores"] = dice_scores
        log_first_n("INFO", f"Dice scores for {len(self.dice_scores)} instances: {dice_scores[:10]}", n=5)
        return results
