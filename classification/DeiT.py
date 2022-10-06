import argparse
import torch
from timm.models import create_model
import timm
assert timm.__version__ == "0.3.2"

import pvt
import pvt_v2
import ours
import ours_channel
import time
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

try:
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000

    torch.cuda.empty_cache()
    FPS = 1000 / latency 
    return latency, FPS

def compute_my_latency():
    """
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    """
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    # model = EfficientNet.from_name('efficientnet-b0')
   
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    input_shape = (1,3,224,224)

    latency, FPS = compute_latency_ms_pytorch(model, input_shape, 1000,  )

    print("The latency of the module is {} ms\n".format(latency))
    print("The FPS of the module is {} ms^(-1)\n".format(FPS))
    
if __name__ == '__main__':
    compute_my_latency()
