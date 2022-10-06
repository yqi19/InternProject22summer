import argparse
import torch
from timm.models import create_model
"""
import pvt
import pvt_v2
import ours
import ours_channel
import ours_window
import ours_window_new
"""
import our_new_window

try:
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Get FLOPS of a classification model')
    parser.add_argument('model', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim

def li_sra_flops(h, w, dim):
    return 2 * h * w * 7 * 7 * dim

def multi_head_self_attention_channel_flops(h,w,dim):
    # pooled features = 4*8 = 32
    assert h*w == 49, "wrong dimention of h*w in stage4!"
    return 2*((16*16)**2)*(h*w)

def global_cross_attention_channel_flops(h,w,dim):
    # 32*16 global features
    # 32 pooled features
    # assert dim == 32*16, "wrong dimention of embedded_dim in stage4!"
    return 2*(dim*16)*h*w

def multi_head_self_attention_flops(h,w,dim,stage):
    if stage == 1:
        return 2*((7*7)**2)*dim
    elif stage == 2:
        return 2*((7*7)**2)*dim
    elif stage == 3:
        return 2*((7*7)**2)*dim
    elif stage == 4:
        return 2*((7*7)**2)*dim

def global_cross_attention_flops(h,w,dim,stage):
    if stage == 1:
        return 2*(h*w*(7*7))*dim
    elif stage == 2:
        return 2*(h*w*(7*7))*dim
    elif stage == 3:
        return 2*(h*w*(7*7))*dim
    elif stage == 4:
        return 2*(h*w*(7*7))*dim

def cross_window_attention_flops(h,w,dim,stage):
    return 2*7*7*(((h*w)//(7*7))*(2*2)*dim)

def cross_window_attention_flops_new(h,w,dim,stage):
    return 2*(h*w)*(1*1)*dim

def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    if 'pvt' in model.name:
        _, H, W = input_shape
        if 'li' in model.name:  # calculate flops of PVTv2_li
            stage1 = li_sra_flops(H // 4, W // 4,
                                  model.block1[0].attn.dim,0) * len(model.block1)
            stage2 = li_sra_flops(H // 8, W // 8,
                                  model.block2[0].attn.dim,1) * len(model.block2)
            stage3 = li_sra_flops(H // 16, W // 16,
                                  model.block3[0].attn.dim,2) * len(model.block3)
            stage4 = li_sra_flops(H // 32, W // 32,
                                  model.block4[0].attn.dim,3) * len(model.block4)
        else:  # calculate flops of PVT/PVTv2
            stage1 = sra_flops(H // 4, W // 4,
                               model.block1[0].attn.sr_ratio,
                               model.block1[0].attn.dim) * len(model.block1)
            stage2 = sra_flops(H // 8, W // 8,
                               model.block2[0].attn.sr_ratio,
                               model.block2[0].attn.dim) * len(model.block2)
            stage3 = sra_flops(H // 16, W // 16,
                               model.block3[0].attn.sr_ratio,
                               model.block3[0].attn.dim) * len(model.block3)
            stage4 = sra_flops(H // 32, W // 32,
                               model.block4[0].attn.sr_ratio,
                               model.block4[0].attn.dim) * len(model.block4)
        flops += stage1 + stage2 + stage3 + stage4
    elif 'ourvisiontransformer' in model.name:
        _,H,W = input_shape
        stage1 = multi_head_self_attention_flops(H//4, W//4, model.block1[0].multiheadselfattn.dim,1)*len(model.block1)
        stage1 += global_cross_attention_flops(H//4, W//4,model.block1[0].crosswindowattn.dim,1)*len(model.block1)

        stage2 = multi_head_self_attention_flops(H/8, W//8, model.block2[0].multiheadselfattn.dim,2)*len(model.block2)
        stage2 += global_cross_attention_flops(H//8, W//8,model.block2[0].crosswindowattn.dim,2)*len(model.block2)

        stage3 = multi_head_self_attention_flops(H//16, W//16, model.block3[0].multiheadselfattn.dim,3)*len(model.block3)
        stage3 += global_cross_attention_flops(H//16, W//16,model.block3[0].crosswindowattn.dim,3)*len(model.block3)

        stage4 = multi_head_self_attention_flops(H//32, W//32, model.block4[0].multiheadselfattn.dim,4)*len(model.block4)
        stage4 += global_cross_attention_flops(H//32, W//32,model.block4[0].crosswindowattn.dim,4)*len(model.block4)
        
        flops += stage1 + stage2 + stage3 + stage4
    elif 'channel' in model.name:
        _, H,W = input_shape
        stage1 = multi_head_self_attention_flops(H//4, W//4, model.block1[0].multiheadselfattn.dim,1)*len(model.block1)
        stage1 += global_cross_attention_flops(H//4, W//4,model.block1[0].crosswindowattn.dim,1)*len(model.block1)

        stage2 = multi_head_self_attention_flops(H/8, W//8, model.block2[0].multiheadselfattn.dim,2)*len(model.block2)
        stage2 += global_cross_attention_flops(H//8, W//8,model.block2[0].crosswindowattn.dim,2)*len(model.block2)

        stage3 = multi_head_self_attention_flops(H//16, W//16, model.block3[0].multiheadselfattn.dim,3)*len(model.block3)
        stage3 += global_cross_attention_flops(H//16, W//16,model.block3[0].crosswindowattn.dim,3)*len(model.block3)

        stage4 = multi_head_self_attention_channel_flops(H//32, W//32, model.block4[0].multiheadselfattn_channel.dim)*len(model.block4)
        stage4 += global_cross_attention_channel_flops(H//32, W//32, model.block4[0].crosswindowattn_channel.dim)*len(model.block4)

        flops += stage1 + stage2 + stage3 + stage4
    elif 'window' in model.name:
        _,H,W = input_shape
        stage1 = multi_head_self_attention_flops(H//4, W//4, model.block1[0].multiheadselfattn.dim,1)*len(model.block1)
        stage1 = cross_window_attention_flops_new(H//4, W//4,model.block1[0].crosswindowattn.dim,1)*len(model.block1)

        stage2 = multi_head_self_attention_flops(H/8, W//8, model.block2[0].multiheadselfattn.dim,2)*len(model.block2)
        stage2 = cross_window_attention_flops_new(H//8, W//8,model.block2[0].crosswindowattn.dim,2)*len(model.block2)

        stage3 = multi_head_self_attention_flops(H//16, W//16, model.block3[0].multiheadselfattn.dim,3)*len(model.block3)
        stage3 = cross_window_attention_flops_new(H//16, W//16,model.block3[0].crosswindowattn.dim,3)*len(model.block3)

        stage4 = multi_head_self_attention_flops(H//32, W//32, model.block4[0].multiheadselfattn.dim,4)*len(model.block4)
        stage4 = cross_window_attention_flops_new(H//32, W//32,model.block4[0].crosswindowattn.dim,4)*len(model.block4)
        
        flops += stage1 + stage2 + stage3 + stage4
    else:
        _,H,W = input_shape

    return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000
    )
    model.name = args.model
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    flops, params = get_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()