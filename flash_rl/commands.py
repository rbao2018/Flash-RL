from dataclasses import asdict
import logging
import argparse
import os 
import yaml

from .configs import get_default_config
from .flash_quantization import profiling_int8, profiling_fp8

logger = logging.getLogger(__name__)

def setup_flashrl(name, parser):
    subparser = parser.add_parser(
        name, 
        description="setup Flash RL", 
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-m', '--model',
        required=False,
        type=str, 
        default=None,
        help='path to the quantized model',
    )
    subparser.add_argument(
        '-p', '--profile',
        required=False,
        type=str, 
        default=None,
        help='path to the profile file',
    )
    subparser.add_argument(
        '--fn', 
        required=False, 
        choices=['fp8', 'fp8_vllm', 'fp8_channel', 'fp8_tensor', 'int8', 'int8_wo_prune', 'int8_prune', 'bf16'],
        default='int8',
        help='quantization function to use',
    )
    subparser.add_argument(
        '-o', '--config-output', 
        required=False, 
        type=str, 
        default=None, 
        help='path to save the config file',
    )
    subparser.add_argument(
        '-a', '--append',
        action='store_true',
        help='append the config to the existing file',
    )    
    subparser.add_argument(
        'columns', 
        nargs=argparse.REMAINDER, 
        help=[
            'other parameters for online quantization, format is'
            'distributed_executor_backend=\"external_launcher\"'
            'module_attribute_to_preserve=[\"workspace\"]'
            'params_to_ignore=[\"q_scale\"]'
        ],
    )
    subparser.set_defaults(func=setup_flashrl_runner)
    return subparser

def setup_flashrl_env():
    import site
    path = site.getsitepackages()[0]
    need_usercustomize = True
    if os.path.exists(os.path.join(path, 'usercustomize.py')):
        with open(os.path.join(path, 'usercustomize.py'), 'r') as f:
            for line in f.readlines():
                if 'import flash_rl' in line and not line.strip().startswith("#"):
                    logger.info("flash_rl already imported in usercustomize.py")
                    need_usercustomize = False
                    break
                           
    if need_usercustomize:
        with open(os.path.join(path, 'usercustomize.py'), 'a') as f:
            f.write(f"try: import flash_rl\nexcept ImportError: pass\n")
            logger.info("flash_rl setup added to usercustomize.py")

def setup_flashrl_runner(args):
    if args.config_output is None:
        logger.info("No config output path provided, using default: ~/.flashrl_config.yaml")
        args.config_output = os.path.expanduser("~/.flashrl_config.yaml")
    
    config_data = asdict(get_default_config(args.fn))
    
    assert args.model is not None or args.fn in ['bf16', 'fp8_vllm'], \
        f"model path is required for quantization {args.fn}"
    
    if args.model is not None:
        config_data['model'] = args.model
    if args.profile is not None:
        config_data['profile'] = args.profile
    config_data['fn'] = args.fn
    for column in args.columns:
        key, value = column.split('=')
        config_data[key] = eval(value)
    
    assert config_data['load_format'] == 'auto' or args.fn in ['fp8', 'fp8_tensor', 'fp8_channel'], \
        f"load_format should be 'auto' for {args.fn}, but got {config_data['load_format']}"
    
    if args.append and os.path.exists(args.config_output):
        with open(args.config_output, 'r') as fin:
            meta_configs = yaml.safe_load(fin)
        meta_configs['configs'].append(config_data) 
    else:
        meta_configs = {'configs': [config_data]}
        
    with open(args.config_output, 'w') as fout:
        yaml.dump(
            meta_configs, 
            fout,
        )
    
    logger.info(f"FlashRL config saved to {args.config_output}")
    setup_flashrl_env()
    
    logger.info(
        f"FlashRL profile saved to {args.config_output}, "
        f"set FLASHRL_CONFIG={args.config_output} to enable it."
    )

def clean_up_flashrl(name, parser):
    subparser = parser.add_parser(
        name, 
        description="setup Flash RL", 
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-p', '--path', 
        required=False, 
        type=str, 
        default=None, 
        help='path to save the config file',
    )
    subparser.set_defaults(func=cleanup_flashrl_runner)
    return subparser

def cleanup_flashrl_runner(args):
    if args.path is None:
        import site
        path = site.getsitepackages()[0]
        if os.path.exists(os.path.join(path, 'usercustomize.py')):
            with open(os.path.join(path, 'usercustomize.py'), 'r') as f:
                lines = f.readlines()
                
            need_write = False 
            for i in range(len(lines)):
                if 'import flash_rl' in lines[i]:
                    l_processed = lines[i].strip()
                    if not l_processed.startswith("#"):
                        if l_processed == 'import flash_rl':
                            lines[i] = f"# {lines[i]}"
                            logger.info("flash_rl setup removed from usercustomize.py")
                            need_write = True
                        elif l_processed == 'try: import flash_rl':
                            lines[i] = f"# {lines[i]}"
                            lines[i+1] = f"# {lines[i+1]}"
                            logger.info("flash_rl setup removed from usercustomize.py")
                            need_write = True
                        else:
                            logger.warning(
                                "flash_rl setup found in usercustomize.py, "
                                "but not removed, due to unknown format."
                                f"please remove the command {lines[i]} manually."
                                f"from {os.path.join(path, 'usercustomize.py')}"
                            )
            
            if need_write:
                with open(os.path.join(path, 'usercustomize.py'), 'w') as f:
                    f.writelines(lines)

def profile_flashrl(name, parser):
    subparser = parser.add_parser(
        name, 
        description="setup Flash RL", 
        help='setup Flash RL',
    )
    subparser.add_argument(
        '-m', '--model', 
        required=False, 
        type=str, 
        help='path to the original model',
    )
    subparser.add_argument(
        '-q', '--quantized',
        required=True,
        type=str,
        help='path to the quantized model',
    )
    subparser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='path to save the profile file',
    )
    subparser.add_argument(
        '--fn', 
        choices=['fp8', 'int8'],
        default='int8',
    )
    subparser.set_defaults(func=profile_runner)
    return subparser

def profile_runner(args):
    if args.fn == 'int8':
        assert args.model is not None, f"model path is required for quantization {args.fn}"
        profiling_int8(args.model, args.quantized, args.output)
    else:
        profiling_fp8(args.quantized, args.output)

def run():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            "setup": setup_flashrl,
            "cleanup": clean_up_flashrl,
            "profile": profile_flashrl,
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand(name, subparsers)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
        