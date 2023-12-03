import torch
import numpy as np

import SAFE.transforms_detr as T

from functools import partial
from PIL import Image

from modeling.DETR.util import box_ops
from modeling.DETR.models import build_model as build_detr_model
from core.detr_args import get_args_parser, set_model_defaults

from torch.autograd import Variable
from torch.nn.functional import sigmoid
from torchvision.transforms.functional import normalize

import os

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes


###############################
#####					  #####
##### 	 PREPROCESSORS	  #####
#####					  #####
###############################

### DETR mapper function
def mapper_func(input_dict, transform):
	name = input_dict['file_name']
	img = Image.open(name).convert('RGB')
	input_dict['image'], _ = transform(img, None)
	return input_dict

def get_preprocess():
	normalize = T.Compose([
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	transform = T.Compose([
		T.RandomResize([800], max_size=1333),
		normalize,
	])

	return transform

def get_mapper():
	return partial(mapper_func, transform=get_preprocess())

def preprocess(img):
	img = img.float()
	img = img / 255.0
	img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return img

@torch.no_grad()
def channel_shift(img):
	return img[[2, 1, 0], :, :]

def modify_voc_dict(mapping_dict):
	siren2vos, _ = get_voc_class_mappers()
	return {i: mapping_dict[k] for i, k in enumerate(siren2vos)}

def get_voc_class_mappers():
	vos_labels = ['person','bird','cat','cow','dog','horse','sheep','airplane','bicycle','boat','bus','car','motorcycle','train','bottle','chair','dining table','potted plant','sofa','tv',]
	# siren_labels = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "dining table", "dog", "horse", "motorcycle", "person","potted plant", "sheep", "sofa", "train", "tv"]

	## Siren labels are already sorted alphabetically.
	## Mapping from VOS to SIREN just requires getting the sorted label ordering
	## and the inverse mapping just requires applying that sorting to an aranged array.
	siren2vos = np.argsort(vos_labels)
	vos2siren = np.argsort(siren2vos)
	
	return siren2vos, vos2siren



###############################
#####					  #####
##### 		BUILDER 	  #####
#####					  #####
###############################

def build_model(args, **kwargs):
	# Build predictor
	
	temp_args = get_args_parser()
	temp_args, _ = temp_args.parse_known_args()
	
	temp_args = set_model_defaults(temp_args)

	if "BDD" in args.config_file:
		temp_args.dataset_file = "bdd"
		temp_args.num_classes = 10
		dset = "bdd"
	else:
		temp_args.dataset_file = "coco"
		temp_args.num_classes = 20
		dset = "voc"

	## Defaults for this codebase
	temp_args.dataset = 'coco_ood_val'
	temp_args.load_backbone = 'dino'
	temp_args.batch_size = 1
	temp_args.eval = True
	
	
	model, criterion, postprocessing = build_detr_model(temp_args)
	
	#print('loading checkpoint...')
	checkpoint = torch.load(os.path.join("./ckpts", "checkpoint_{dset}_vanilla.pth"), map_location='cpu')
	missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
	model.cuda()
	model.eval()


	# print(model)
	# print(f"Missing: {missing_keys}")
	# print(f"Unexpected: {unexpected_keys}")

	return model, criterion, postprocessing




###############################
#####					  #####
##### 		INFERENCE	  #####
#####					  #####
###############################

@torch.no_grad()
def forward(predictor, input_img, postprocessors):
	## Short form variables for neatness
	image, h, w = [input_img[0][key] for key in ['image', 'height', 'width']]
	
	#print(image.size())
	## Perform forward pass over the input image
	outputs = predictor(image.to(0).unsqueeze(0))
	
	## Apply postprocessing to the detections as part of DETR pipeline
	outs = postprocessors['bbox'](outputs, torch.Tensor([h, w]).unsqueeze(0).cuda())[0]

	## Placeholders to shorten conversion code further down
	n_boxes = len(outs['boxes'])
	covars = [torch.diag(torch.Tensor([2.1959e-5, 2.1973e-5]*2)) for _ in range(n_boxes)]

	## SIREN model outputs are in a different format to what is expected by VOS.
	## This section converts the model outputs to the correct format for VOS.
	out_dict = {
		'pred_boxes': Boxes(outs['boxes']), 
		'scores': outs['scores'], 
		'pred_classes': outs['labels'],
		'pred_cls_probs': sigmoid(outs['logits_for_ood_eval'][0]),
		'inter_feat': outs['logits_for_ood_eval'][0], 
		'pred_boxes_covariance': torch.stack(covars).cuda() if n_boxes else torch.Tensor([]).cuda(), 
		'logistic_score': torch.zeros(n_boxes) 
	}

	## Finalise conversion by creating instance objects with the appropriate fields.
	outputs = Instances(image_size=(h, w), **out_dict)

	## Return the newly formatted outputs, the predicted regions of interest (for SAFE) and a skip signifier
	return outputs, outs['boxes'], n_boxes < 1




###############################
#####					  #####
##### 		TRANSFORM	  #####
#####					  #####
###############################

def fgsm(inputs, model, crit, eps=8.0):
	## Grab a copy of the original image for gradient processing
	grad_img = inputs[0]['image'].clone().detach().float().cuda()

	## Track gradients w.r.t input image
	grad_img = Variable(grad_img, requires_grad=True)

	## Image preprocessing
	grad_img = preprocess(grad_img)

	## Sanity checks
	assert model.training
	model.zero_grad()

	## Forward pass the input
	outputs = model([grad_img])

	## DETR outputs for VOC do not correspond to the same ordering as VOS benchmark expects.
	## Convert to correct label ordering if the target dataset is VOC.
	labels = inputs[0]['instances'].gt_classes
	

	if outputs['pred_logits'].size(-1) > 10: 
		cls_mapper=get_voc_class_mappers()[1]
		labels = torch.from_numpy(cls_mapper)[labels]

	labels = labels.cuda()

	## Configure the gt boxes
	boxes = box_ops.box_xyxy_to_cxcywh(inputs[0]['instances'].gt_boxes.tensor.cuda())
	boxes = torch.stack([
		boxes[:, 0]/inputs[0]['width'],
		boxes[:, 1]/inputs[0]['height'],
		boxes[:, 2]/inputs[0]['width'],
		boxes[:, 3]/inputs[0]['height'],
	], dim=1)
		
	## Setup the targets
	targets = [{
		'labels': labels,
		'boxes': boxes
	}]
	
	## Compute the weighted loss on the target image
	loss_dict = crit(outputs, targets)
	weight_dict = crit.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

	## Compute the gradient with respect to the input image
	grad = torch.autograd.grad(losses, grad_img, retain_graph=False, create_graph=False)[0]

	## Remove gradients for sanity
	model.zero_grad()

	## Generate perturbed clone image
	new_img = inputs[0]['image'].clone().detach().cuda() 
	new_img = new_img + eps*grad.sign()
	new_img = torch.clamp(new_img, min=0, max=255).detach()
	new_img = preprocess(new_img)

	## Return new image
	return new_img