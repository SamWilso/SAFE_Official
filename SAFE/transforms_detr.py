#from detectron2.utils.events import EventStorage
import torch
from torch.autograd import Variable
from torchvision.transforms.functional import normalize

VOS_VOC_CLASS_ORDERING = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','couch','tv',]
SIREN_VOC_CLASS_ORDERING = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "couch", "train", "tv"]

SIREN_VOC_IDS = {k:i for i, k in enumerate(SIREN_VOC_CLASS_ORDERING)}
VOS_VOC_IDS = {k:i for i, k in enumerate(VOS_VOC_CLASS_ORDERING)}
VOC_VOS2SIREN_IDS = {VOS_VOC_IDS[a]: SIREN_VOC_IDS[a] for a in SIREN_VOC_CLASS_ORDERING}
voc_vos2siren_mapping = lambda a: VOC_VOS2SIREN_IDS[a.item()]


def fgsm(inputs, model, crit, postprocessors, eps=8):
	eps /= 255.0
	grad_img = inputs[0]['image'].detach()
	grad_img = Variable(grad_img.float().cuda(), requires_grad=True)
	raw_scaled_ = grad_img / 255.0
	raw_scaled_ = raw_scaled_[[2, 1, 0], :, :]
	normed_ = normalize(raw_scaled_, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
	#normed_ = torch.zeros_like(normed_)
	
	model.train()
	model.zero_grad()

	outputs = model([normed_.clone()])
	#print(outputs)
	inst = inputs[0]['instances'] 

	# VOC vs BDD check
	if outputs['pred_logits'].size(-1) > 10: 
		labels = torch.Tensor(list(map(voc_vos2siren_mapping, inst.gt_classes.detach()))).cuda().long()
	else:
		labels = inst.gt_classes.cuda()
	
	from .util import box_ops
	boxes = box_ops.box_xyxy_to_cxcywh(inst.gt_boxes.tensor.cuda())
	boxes = torch.stack([
		boxes[:, 0]/inputs[0]['width'],
		boxes[:, 1]/inputs[0]['height'],
		boxes[:, 2]/inputs[0]['width'],
		boxes[:, 3]/inputs[0]['height'],
	], dim=1)

	targets = [{
		'labels': labels,#inst.gt_classes.cuda(),
		'boxes': boxes
	}]
	loss_dict = crit(outputs, targets)
	weight_dict = crit.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
	

	grad = torch.autograd.grad(losses, grad_img, retain_graph=False, create_graph=False)[0]

	model.zero_grad()
	new_img = inputs[0]['image'].clone().detach().cuda() / 255.0
	
	new_img = new_img + eps*grad.sign()
	new_img = torch.clamp(new_img, min=0, max=1.0).detach()

	return new_img

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random
import torchvision.transforms.functional_tensor as TF

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "patches"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target



def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target




class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)



class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

