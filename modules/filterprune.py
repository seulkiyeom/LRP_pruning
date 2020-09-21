#For searching based on criteria
import torch
import torch.nn as nn

from heapq import nsmallest
from operator import itemgetter

EXTREMELY_HIGH_VALUE = 99999999

def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data

def save_gradient(self, grad_input, grad_output):
    self.grad = grad_output[0]

class FilterPrunner:
    def __init__(self, model, args):
        self.model = model
        self.reset()
        self.args = args

    def reset(self):
        self.filter_ranks = {}
        # self.forward_hook()

    def forward_hook(self):
        # For Forward Hook
        for name, module in self.model.named_modules():
            module.register_forward_hook(fhook)
        # for name, module in self.model.classifier.named_modules():
        #     module.register_forward_hook(fhook)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_backward_hook(save_gradient)

    def compute_filter_criterion(self, layer_type='conv', relevance_method='z',
                                 criterion='lrp'):
        for name, module in self.model.named_modules():
            if criterion == 'lrp' or criterion == 'weight':
                if hasattr(module, "module"):
                    if isinstance(module.module,
                                  nn.Conv2d) and layer_type == 'conv':
                        if name not in self.filter_ranks:
                            self.filter_ranks[name] = torch.zeros(
                                module.relevance.shape[1])
                        # else:
                        #     print("here")

                        if criterion == 'lrp':
                            values = torch.sum(module.relevance.abs(), dim=(0, 2, 3))
                        elif criterion == 'weight':
                            values = torch.sum(module.module.weight.abs(), dim=(1, 2, 3))

                        if hasattr(module, "output_mask"):
                            values[module.output_mask == 0] = EXTREMELY_HIGH_VALUE
                        self.filter_ranks[name] += values.cpu() if torch.cuda.is_available() else values
            elif criterion == 'grad' or criterion == 'ICLR':
                if isinstance(module,
                              nn.Conv2d) and layer_type == 'conv':
                    if name not in self.filter_ranks:
                        self.filter_ranks[name] = torch.zeros(
                            module.output.shape[1])

                    if criterion == 'grad':
                        values = torch.sum(module.grad, dim=(0, 2, 3))
                        values = \
                            values / (module.output.size(0) * module.output.size(2) * module.output.size(3))
                    elif criterion == 'ICLR':
                        values = torch.sum((module.output * module.grad), dim=(0, 2, 3))
                        values = \
                            values / (module.output.size(0) * module.output.size(2) * module.output.size(3))
                    if hasattr(module, "output_mask"):
                        values[module.output_mask == 0] = EXTREMELY_HIGH_VALUE

                    self.filter_ranks[name] += values.cpu() if torch.cuda.is_available() else values

    def forward(self, x):
        self.activations = []  # 전체 conv_layer의 activation map 수
        self.weights = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}  # conv layer의 순서 7: 17의미는 7번째 conv layer가 전체에서 17번째에 있다라는 뜻

        activation_index = 0
        for layer, (name, module) in enumerate(
                self.model.features._modules.items()):
            x = module(x)  # 일반적인 forward를 수행하면서..
            if isinstance(module,
                          torch.nn.modules.conv.Conv2d):  # conv layer 일때 여기를 지나감
                x.register_hook(self.compute_rank)
                if self.args.method_type == 'weight':
                    self.weights.append(module.weight)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(
            self.activations) - self.grad_index - 1  # 뒤에서부터 하나씩 끄집어 냄
        activation = self.activations[activation_index]

        if args.method_type == 'ICLR':
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0,
                0].data  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(
                    2) * activation.size(3))

        elif args.method_type == 'grad':
            values = \
                torch.sum((grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0,
                0].data  # # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(
                    2) * activation.size(3))

        elif args.method_type == 'weight':
            weight = self.weights[activation_index]
            values = \
                torch.sum((weight).abs(), dim=1, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[:, 0, 0,
                0].data  # Many publications based on weight and activation(=feature) map

        else:
            raise ValueError('No criteria')

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(
                activation.size(
                    1)).zero_().cuda() if args.cuda else torch.FloatTensor(
                activation.size(1)).zero_()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            if self.args.norm:  # L2-norm for global rescaling
                if self.args.method_type == 'lrp': #it doesn't need
                    v = self.filter_ranks[i]
                    v = v / torch.sum(v[v < EXTREMELY_HIGH_VALUE])
                    self.filter_ranks[i] = v.cpu()
                elif self.args.method_type == 'weight':  # weight & L1-norm (Li et al., ICLR 2017)
                    v = self.filter_ranks[i]
                    v = v / torch.sum(v[v < EXTREMELY_HIGH_VALUE])  # torch.sum(v) = total number of dataset
                    self.filter_ranks[i] = v.cpu()
                elif self.args.method_type == 'ICLR':  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                    v = torch.abs(self.filter_ranks[i])
                    v = v / torch.sqrt(torch.sum(v[v < EXTREMELY_HIGH_VALUE] * v[v < EXTREMELY_HIGH_VALUE]))
                    self.filter_ranks[i] = v.cpu()
                elif self.args.method_type == 'grad':  # |grad| & L2-norm (Sun et al., ICML 2017)
                    v = torch.abs(self.filter_ranks[i])
                    v = v / torch.sqrt(torch.sum(v[v < EXTREMELY_HIGH_VALUE] * v[v < EXTREMELY_HIGH_VALUE]))
                    self.filter_ranks[i] = v.cpu()
            else:
                if self.args.method_type == 'weight':  # weight
                    v = self.filter_ranks[i]
                    self.filter_ranks[i] = v.cpu()
                elif self.args.method_type == 'ICLR':  # |grad*act|
                    v = torch.abs(self.filter_ranks[i])
                    self.filter_ranks[i] = v.cpu()
                elif self.args.method_type == 'grad':  # |grad|
                    v = torch.abs(self.filter_ranks[i])
                    self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        ranked_filters = self.lowest_ranking_filters(num_filters_to_prune)
        assert len(ranked_filters) == num_filters_to_prune
        assert len(set([x[:2] for x in ranked_filters])) == num_filters_to_prune

        return [x[:2] for x in ranked_filters]

    def lowest_ranking_filters(self, num):
        data = []
        # Make sure that the filter ranks are unique
        assert len(self.filter_ranks) == len(set(self.filter_ranks))
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))
                # data 변수에 모든 layer의 모든 filter의 값을 쭈욱 나열 시킨다.
        # Make sure each filter has only one rank
        assert len(set([x[:2] for x in data])) == len(data)

        filters_to_prune = nsmallest(num, data, itemgetter(
            2))  # data list 내에서 가장 작은 수를 num(=512개) 만큼 뽑아서 리스트에 저장
        # Make sure the filters are unique
        assert len(set([x[:2] for x in filters_to_prune])) == num
        return filters_to_prune