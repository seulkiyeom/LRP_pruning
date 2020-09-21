import torch
from torch.nn.utils import prune as torch_prune

def prune_conv_layer(model, layer_index, filter_index, criterion = 'lrp', cuda_flag = False):
    ''' input parameters
    1. model: 현재 모델
    2. layer_index: 자르고자 하는 layer index
    3. filter_index: 자르고자 하는 layer의 filter index
    '''

    conv = dict(model.named_modules())[layer_index]

    if not hasattr(conv, "output_mask"):
        # Instantiate output mask tensor of shape (num_output_channels, )
        conv.output_mask = torch.ones(conv.weight.shape[0])

    # Make sure the filter was not pruned before
    assert conv.output_mask[filter_index] != 0

    conv.output_mask[filter_index] = 0

    mask_weight = conv.output_mask.view(-1, 1, 1, 1).expand_as(
        conv.weight)
    torch_prune.custom_from_mask(conv, "weight", mask_weight)

    if conv.bias is not None:
        mask_bias = conv.output_mask
        torch_prune.custom_from_mask(conv, "bias", mask_bias)

    if cuda_flag:
        conv.weight = conv.weight.cuda()
        # conv.module.bias = conv.module.bias.cuda()

    return model