def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.0**9, 2)) + "GMac"
    elif flops // 10**6 > 0:
        return str(round(flops / 10.0**6, 2)) + "MMac"
    elif flops // 10**3 > 0:
        return str(round(flops / 10.0**3, 2)) + "KMac"
    return str(flops) + "Mac"


def get_model_parameters_number(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    if params_num // 10**6 > 0:
        return str(round(params_num / 10**6, 2)) + "M"
    elif params_num // 10**3:
        return str(round(params_num / 10**3, 2)) + "k"

    return str(params_num)


def flops_to_string_value(flops):
    return round(flops / 10.0**9, 2)


def get_model_parameters_number_value_mask(model, as_string=False):
    import torch
    import torch.nn as nn

    # params_num = sum(p.numel() for p in model.parameters())
    prev_output_mask = None
    conv_count = 0
    lin_count = 0
    norm_count = 0
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            norm_count += sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.Linear):
            if hasattr(mod, "adapter") and hasattr(mod.adapter, "rank"):  # SPLoRA
                lin_count += mod.adapter.rank * sum(mod.weight.shape)
            else:
                lin_count += sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.Conv2d):
            if hasattr(mod, "output_mask"):
                n_out_channels = len(torch.where(mod.output_mask == 1)[0])
                n_in_channels = mod.in_channels
                if (
                    prev_output_mask is not None
                    and prev_output_mask.shape[0] == mod.in_channels
                ):
                    n_in_channels = len(torch.where(prev_output_mask == 1)[0])
                if hasattr(mod, "adapter"):
                    if hasattr(mod.adapter, "rank"):  # SPLoRA
                        conv_count += mod.adapter.rank * (
                            n_out_channels + n_in_channels
                        )
                    else:  # SPPaRA
                        conv_count += n_out_channels * n_in_channels
                else:
                    conv_count += (
                        n_out_channels
                        * n_in_channels
                        * mod.weight.size(2)
                        * mod.weight.size(3)
                    )
                prev_output_mask = mod.output_mask
            else:
                conv_count += sum(p.numel() for p in mod.parameters())

    params_num = conv_count + lin_count + norm_count

    if not as_string:
        return params_num

    return round((params_num) / 10**6, 2)


def get_model_parameters_number_value(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    return round(params_num / 10**6, 2)
