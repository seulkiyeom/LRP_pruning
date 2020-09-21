def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + 'GMac'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + 'MMac'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + 'KMac'
    return str(flops) + 'Mac'


def get_model_parameters_number(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + 'M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + 'k'

    return str(params_num)

def flops_to_string_value(flops):
    return round(flops / 10.**9, 2)


def get_model_parameters_number_value_mask(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters())
    # params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    del_param = 0
    import torch.nn as nn
    import torch
    for name, module in model.named_modules():
        if hasattr(module, "module"): #lrp and weight
            if isinstance(module.module, nn.Conv2d):
                del_param += module.module.bias.size(0) #need to always remove bias term
                if hasattr(module, "output_mask"):
                    n_filter = len(torch.where(module.output_mask == 0)[0])
                    # print(f'{name}: {n_filter}')
                    del_param += n_filter * module.module.weight.size(1) * module.module.weight.size(2) * module.module.weight.size(3)


        elif isinstance(module, nn.Conv2d): #grad and ICLR
            if hasattr(module, "output_mask"):
                n_filter = len(torch.where(module.output_mask == 0)[0])
                # print(f'{name}: {n_filter}')
                del_param += n_filter * module.weight.size(1) * module.weight.size(2) * module.weight.size(3)

    if not as_string:
        return params_num - del_param

    return round((params_num - del_param)/ 10 ** 6, 2)

def get_model_parameters_number_value(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    return round(params_num / 10 ** 6, 2)