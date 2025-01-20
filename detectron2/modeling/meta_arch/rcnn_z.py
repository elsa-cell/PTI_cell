import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..separators import build_separator


from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN_Z"]

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_Z(GeneralizedRCNN):
    """
    Generalized R-CNN_Z. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.

        Args:
            cfg: the desired configuration
        """
        #super().__init__(cfg)
        self._image_dim = cfg.MODEL.BACKBONE.IMAGE_DIM
        if self._image_dim == 2:
            input_shape = ShapeSpec(channels = cfg.INPUT.STACK_SIZE * len(cfg.MODEL.PIXEL_MEAN))
        elif self._image_dim == 3:
            input_shape = ShapeSpec(stack_size=cfg.INPUT.STACK_SIZE, channels=len(cfg.MODEL.PIXEL_MEAN))



        backbone = build_backbone(cfg, input_shape=input_shape)

        super().__init__(
            backbone = backbone,
            proposal_generator = build_proposal_generator(cfg, backbone.output_shape()),
            roi_heads = build_roi_heads(cfg, backbone.output_shape()),
            pixel_mean = cfg.MODEL.PIXEL_MEAN,
            pixel_std = cfg.MODEL.PIXEL_STD,
            input_format = cfg.INPUT.FORMAT,
            vis_period = cfg.VIS_PERIOD
        )

        assert(cfg.DATALOADER.IS_STACK == True)

        
        self.pixel_mean = self.pixel_mean.to(self.device)
        self.pixel_std  = self.pixel_std.to(self.device)

        self.separator = build_separator(cfg, self.backbone.output_shape())
        self._stack_size = cfg.INPUT.STACK_SIZE        
        

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list (nb_stacks_per_batch) of a list (nb_img_per_stack), batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        
        Does : For a stack of size self._stack_size (in config):
            inference if not test (see dedicated function)
            test :
                1. Preprocess the images
                2. Reshape the tensor to be in format (N, Cin, H, W)
                3. Goes through backbone = feature extraction
                4. Feature separation to have one set of features for each representation of an image of the stack
                5. Goes through the proposal generator
                6. Goes through the roi heads
                7. Update losses
        """
        if not self.training:
            return self.inference(batched_inputs)

        nb_stacks = len(batched_inputs)

        original_stacks = [None] * nb_stacks
        for s in range(nb_stacks):
            original_stacks[s] = [x["image"].to(self.device) for x in batched_inputs[s]]

        ## 1. Preprocess the images
        # Shape of tensor : (C, H, W)
        # NB : stacks_norm corresponds to stacks_norm in AugP-creatis/AdelaiDet-Z repo, and to images in detectron2 repo
        #      preprocess_image corresponds to normalizer in condinst_z in AugP-creatis/AdelaiDet-Z repo, and to preprocess_image in detectron2 repo
        stacks_norm = [None] * nb_stacks
        for s in range(nb_stacks):
            stacks_norm[s] = torch.stack([self.preprocess_image(x) for x in original_stacks[s]], dim=1)
            # Shape of tensor : (C, Z, H, W)
        stacks_norm = ImageList.from_tensors(stacks_norm, self.backbone.size_divisibility)
        # Shape of tensor : (N, C, Z, H, W)


        if "instances" in batched_inputs[0][0]:
            stack_gt_instances = [None] * nb_stacks
            z_gt_instances = [[None] * nb_stacks for z in range(self._stack_size)]
            for s in range(nb_stacks):
                stack_gt_instances[s] = [x["instances"].to(self.device) for x in batched_inputs[s]]
                for z in range(self._stack_size):
                    z_gt_instances[z][s] = stack_gt_instances[s][z]
        else:
            stack_gt_instances = [None] * nb_stacks
            z_gt_instances = [None] * self._stack_size


        if self._image_dim == 2:
            ## 2. Reshape the tensor to be in format (N, Cin, H, W)
            tensor_size = stacks_norm.tensor.shape                                                           
            # Shape of tensor : (N, C, Z, H, W)
            input_tensor = stacks_norm.tensor.view(tensor_size[0], -1, tensor_size[-2], tensor_size[-1])    
            # Shape of tensor : (N, C*Z, H, W)
        elif self._image_dim == 3:
            input_tensor = stacks_norm.tensor
            # Shape of tensor : (N, C, Z, H, W)

        ## 3. Goes through backbone = feature extraction
        features = self.backbone(input_tensor)         # Backbone 2D takes (N, CHANNELS, H, W), with (N, C*Z, H, W) ; 3D takes (N, C, Z, H, W)
        # Shape of tensor : (N, Cout, H, W)

        ## 4. Feature separation to have one set of features for each representation of an image of the stack
        z_features = self.separator(features)   
        # List (len = Z) of tensor : (N, Cout, H, W)



        ## 5. Goes through the proposal generator
        ## 6. Goes through the roi heads
        ## 7. Update losses
        losses = {}
        for z in range(self._stack_size):
            if self.proposal_generator:
                proposals, proposal_losses = self.proposal_generator(stacks_norm, z_features[z], z_gt_instances[z])
            else:
                assert "proposals" in batched_inputs[0][0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs[z]]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(stacks_norm, z_features[z], proposals, z_gt_instances[z])
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs[z], proposals)

            # On garde en mémoire les losses de toutes les images
            proposal_losses_keys = list(proposal_losses.keys()).copy()
            for l in proposal_losses_keys:
                proposal_losses["{}_{}".format(z, l)] = proposal_losses.pop(l)

            detector_losses_keys = list(detector_losses.keys()).copy()
            for l in detector_losses_keys:
                detector_losses["{}_{}".format(z, l)] = detector_losses.pop(l)

            
            losses.update(detector_losses)
            losses.update(proposal_losses)

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.

        Does:
            For a stack of size self._stack_size (in config):
            1. Preprocessing (same for every input) : Normalize, pad and batch the input images.
            2. Reshape the tensor to be in format (N, Cin, H, W)
            3. Goes through backbone = feature extraction
            4. Feature separation to have one set of features for each representation of an image of the stack
            5. Goes through the proposal generator
            6. Goes through the roi heads
            7. Postprocessing if needed, (same for every input) : Rescale the output instances to the target size.
        """
        assert not self.training

        nb_stacks = len(batched_inputs)

        original_stacks = [None] * nb_stacks
        for s in range(nb_stacks):
            original_stacks[s] = [x["image"].to(self.device) for x in batched_inputs[s]]


        ## 1. Preprocess the images
        # Shape of tensor : (C, H, W)
        stacks_norm = [None] * nb_stacks
        for s in range(nb_stacks):
            stacks_norm[s] = torch.stack([self.preprocess_image(x) for x in original_stacks[s]], dim=1)
            # Shape of tensor : (C, Z, H, W)
        stacks_norm = ImageList.from_tensors(stacks_norm, self.backbone.size_divisibility)
        # Shape of tensor : (N, C, Z, H, W)

        if self._image_dim == 2:
            ## 2. Reshape the tensor to be in format (N, Cin, H, W)
            tensor_size = stacks_norm.tensor.shape                                                           
            # Shape of tensor : (N, C, Z, H, W)
            input_tensor = stacks_norm.tensor.view(tensor_size[0], -1, tensor_size[-2], tensor_size[-1])    
            # Shape of tensor : (N, C*Z, H, W)
        elif self._image_dim == 3:
            input_tensor = stacks_norm.tensor
            # Shape of tensor : (N, C, Z, H, W)


        ## 3. Goes through backbone = feature extraction
        features = self.backbone(input_tensor)         # Backbone 2D takes (N, CHANNELS, H, W), with (N, C*Z, H, W) ; 3D takes (N, C, Z, H, W)
        # Shape of tensor : (N, Cout, H, W)

        ## 4. Feature separation to have one set of features for each representation of an image of the stack
        z_features = self.separator(features)   
        # List (len = Z) of tensor : (N, Cout, H, W)


        # 5. Goes through the proposal generator
        # THEN
        # 6. Goes through the roi heads
        # According to several cases detailed in the if else
        results = [None] * self._stack_size
        if detected_instances is None:
            for z in range(self._stack_size):
                if self.proposal_generator:
                    proposals, _ = self.proposal_generator(stacks_norm, z_features[z], None)
                else:
                    assert "proposals" in batched_inputs[0][0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs[0]]
                results[z], _ = self.roi_heads(stacks_norm, z_features[z], proposals, None)
        else:
            for z in range(self._stack_size):
                detected_instances = [x.to(self.device) for x in detected_instances[z]]
                results[z] = self.roi_heads.forward_with_given_boxes(z_features[z], detected_instances[z])


        # 7. Postprocessing if needed, by image of the stack
        # _postprocess function needs second argument to be a list of the number of images in the batch
        # It corresponds to batched_inputs in the rcnn.py file as it is a list of dict of len=batch_size 
        # Here, batched_inputs is a list (len=batch_size) of a list (len=stack_size) of dict
        # We now want to create a list (len=stack_size) of a list (len=batch_size) of dict 
        # We will call this list z_batched_inputs
        # so that we can give the list z_batched_inputs[z] to the function _postprocess
        # We will call the function stack_size times
        # NB: z_batched_inputs[z] is of len batch_size, as we need it to be
        z_batched_inputs = [[None] * nb_stacks for z in range(self._stack_size)]
        for s in range(nb_stacks):
            for z in range(self._stack_size):
                z_batched_inputs[z][s] = batched_inputs[s][z]

        if do_postprocess:
            postprocessed_results = [None] * self._stack_size
            for z in range(self._stack_size):
                postprocessed_results[z] = super()._postprocess(results[z], z_batched_inputs[z], stacks_norm.image_sizes)
            return postprocessed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        return (batched_inputs - self.pixel_mean) / self.pixel_std