from torch.nn import Parameter

import torch


class Dropout(torch.nn.Linear):

    def __init__(self, drop_p, layers_size, l,  *args, **kwargs):
        super().__init__(in_features=int(layers_size[l]), out_features=int(layers_size[l+1]))
        weights = ['weight']
        weight_dropout = drop_p
        _weight_drop(self, weights, weight_dropout)


def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, Parameter(w * (1 - dropout)))

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)