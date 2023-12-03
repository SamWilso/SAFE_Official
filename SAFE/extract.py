import torch
import h5py
from tqdm import tqdm
import os
"""
Probabilistic Detectron Inference Script
"""
import core
import sys


from tqdm import tqdm

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.engine import launch
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers.distributed_sampler import InferenceSampler

# Project imports
#from core.evaluation_tools.evaluation_utils import get_train_contiguous_id_to_test_thing_dataset_id_dict
from core.setup import setup_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
	if "val" in args.test_dataset: 
		raise ValueError('Error: Feature extraction should only be performed on the training dataset to avoid accidental "training-on-test" errors.')

	##Argument-defined values
	valid_transforms = ['fgsm']
	assert args.transform in valid_transforms, f'Error: Invalid value encountered in "transform" argument. Expected one of: {valid_transforms}. Got: {args.transform}'
	assert args.transform_weight >= 0 and args.transform_weight <= 255,  f'Error: Invalid value encountered in "transform_weight" argument. Expected: 0 <= transform_weight <=255. Got: {args.transform}'


	# Setup config
	cfg = setup_config(args,
					   random_seed=args.random_seed,
					   is_testing=True)
	
	# Make sure only 1 data point is processed at a time. This simulates
	# deployment.
	cfg.defrost()
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.SOLVER.IMS_PER_BATCH = 1

	cfg.MODEL.DEVICE = device.type

	# Set up number of cpu threads#
	torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)
	
	test_data_loader = build_detection_test_loader(
		cfg, dataset_name=args.test_dataset)
	cfg.INPUT.MIN_SIZE_TRAIN=800
	cfg.INPUT.RANDOM_FLIP='none'

	if "RCNN" in args.variant:
		from . import RCNN as model_utils
	else:
		from . import DETR as model_utils


	predictor, criterion, postprocessor = model_utils.build_model(
		cfg=cfg, 
		args=args
	)

	test_data_loader = build_detection_train_loader(
		cfg, 
		sampler=InferenceSampler(len(test_data_loader))
	)    

	from .shared.tracker import featureTracker
	ConvTracker = featureTracker(predictor, args.variant)

	dset = "VOC" if "VOC" in args.config_file else "BDD"
	id_fname = f'{dset}-{args.variant}-standard.hdf5'
	ood_fname = f'{dset}-{args.variant}-{args.transform}-{args.transform_weight}.hdf5'

	tmp_path = os.path.join(args.dataset_dir, "safe")
	if not os.path.exists(tmp_path):
		os.makedirs(tmp_path)
	id_path = os.path.join(tmp_path, id_fname)
	ood_path = os.path.join(tmp_path, ood_fname)


	capture_fn(
		dataloader=test_data_loader,
		model_utils=model_utils,
		predictor=predictor,
		tracker=ConvTracker,
		files=(id_path, ood_path),
		postprocessors=postprocessor,
		criterion=criterion,
		weight=args.transform_weight
	)
	

def capture_fn(dataloader, model_utils, predictor, tracker, files, postprocessors, criterion, weight):
	id_file, ood_file = h5py.File(files[0], 'w'), h5py.File(files[-1], 'w')
	
	for idx, input_im in enumerate(tqdm(dataloader)):
		
		#exit()
		input_im[0]['image'] = model_utils.channel_shift(input_im[0]['image'])
		copy_img = input_im[0]['image'].clone().detach()

		kept_rois = extract_pass(
			input_im=input_im,
			predictor=predictor,
			postprocessors=postprocessors,
			model_utils=model_utils,
			tracker=tracker,
			dset_file=id_file,
			index=idx, 
			kept_rois=None
		)
		
		## Perturbing phase
		mdl = predictor if 'DETR' in files[0] else predictor.model
		mdl.train()

		input_im[0]['image'] = copy_img

		transform_data = {
			'inputs': input_im,
			'model': mdl,
			'crit': criterion,
			'eps': weight
		}
			
		input_im[0]['image'] = model_utils.fgsm(**transform_data)
		
		## End perturbing phase
		mdl.eval()

		_ = extract_pass(
			input_im=input_im,
			predictor=predictor,
			postprocessors=postprocessors,
			model_utils=model_utils,
			tracker=tracker,
			dset_file=ood_file,
			index=idx, 
			kept_rois=kept_rois
		)
		
	id_file.close()
	ood_file.close()
			

@torch.no_grad()
def extract_pass(input_im, predictor, postprocessors, model_utils, tracker, dset_file, index, kept_rois=None):
	h = input_im[0]['height']
	input_im[0]['image'] = model_utils.preprocess(input_im[0]['image']) if kept_rois is None else input_im[0]['image']
	_, boxes, _ = model_utils.forward(
		predictor=predictor,
		input_img=input_im,
		postprocessors=postprocessors,
	)
	kept_rois = boxes if kept_rois is None else kept_rois
	features = tracker.roi_features([kept_rois], h)
	features = features.detach().cpu().numpy()

	dset_file.create_dataset(f'{index}', data=features)
	
	tracker.flush_features()
	return kept_rois




def interface(args):
	print(args)
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)