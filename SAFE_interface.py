import importlib

def add_safe_args(arg_parser):
	## SAFE-specific arguments
	arg_parser.add_argument('--task', 			type=str, default="eval")
	arg_parser.add_argument('--tdset',			type=str, default='VOC')
	arg_parser.add_argument('--bbone',			type=str, default='RN50')
	arg_parser.add_argument('--variant', 		type=str, default="RCNN")
	arg_parser.add_argument('--mlp-path',		type=str, default="")
	arg_parser.add_argument("--transform", 		type=str, default="fgsm")
	arg_parser.add_argument('--transform-weight', type=int, default=8)

	return arg_parser

if __name__ == "__main__":
	from core.setup import setup_arg_parser
	arg_parser = setup_arg_parser()
	arg_parser = add_safe_args(arg_parser)
	args = arg_parser.parse_args()

	regnet_filler = "regnetx_" if args.bbone == "RGX4" else ""
	mode = "val" if args.task.lower() == "eval" else "train"

	args.config_file = f'{args.tdset.upper()}-Detection/faster-rcnn/{regnet_filler}vanilla.yaml'
	args.test_dataset = f"{args.tdset.lower()}_custom_{mode}"
	#args = quick_args(args)
	if args.variant == 'RCNN': args.variant = f'{args.variant}-{args.bbone}'

	if args.tdset == 'VOC':
		args.dataset_dir = f'{args.dataset_dir}VOC_0712_converted/'
	else:
		args.dataset_dir = f'{args.dataset_dir}bdd100k/'

	print("Command Line Args:", args)

	task_module = importlib.import_module(f'SAFE.{args.task.lower()}')
	task_module.interface(args)


	
	