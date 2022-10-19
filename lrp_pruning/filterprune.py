from heapq import nsmallest
from operator import itemgetter

# For searching based on criteria
import torch
import torch.nn as nn

EXTREMELY_HIGH_VALUE = 99999999


def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data


def save_gradient(self, grad_input, grad_output):
    self.grad = grad_output[0]


class FilterPruner:
    def __init__(self, model, args):
        self.set_model(model)
        self.reset()
        self.args = args

    def set_model(self, model):
        self.model = model
        self.forward_hook()

    def reset(self):
        self.filter_ranks = {}
        # self.forward_hook()

    def forward_hook(self):
        # For Forward Hook
        for _, module in self.model.named_modules():
            module.register_forward_hook(fhook)
            if isinstance(module, nn.Conv2d):
                module.register_backward_hook(save_gradient)

    def compute_filter_criterion(  # noqa: C901
        self, layer_type="conv", relevance_method="z", criterion="lrp"
    ):
        for name, module in self.model.named_modules():
            if criterion == "lrp" or criterion == "weight":
                if hasattr(module, "module"):
                    if isinstance(module.module, nn.Conv2d) and layer_type == "conv":
                        if name not in self.filter_ranks:
                            self.filter_ranks[name] = torch.zeros(
                                module.relevance.shape[1]
                            )

                        if criterion == "lrp":
                            values = torch.sum(module.relevance.abs(), dim=(0, 2, 3))
                        elif criterion == "weight":
                            values = torch.sum(
                                module.module.weight.abs(), dim=(1, 2, 3)
                            )

                        if hasattr(module.module, "output_mask"):
                            values[
                                module.module.output_mask == 0
                            ] = EXTREMELY_HIGH_VALUE
                        self.filter_ranks[name] += (
                            values.cpu() if torch.cuda.is_available() else values
                        )
            elif criterion == "grad" or criterion == "taylor":
                if isinstance(module, nn.Conv2d) and layer_type == "conv":
                    if name not in self.filter_ranks:
                        self.filter_ranks[name] = torch.zeros(module.output.shape[1])

                    if criterion == "grad":
                        values = torch.sum(module.grad, dim=(0, 2, 3))
                        values = values / (
                            module.output.size(0)
                            * module.output.size(2)
                            * module.output.size(3)
                        )
                    elif criterion == "taylor":
                        values = torch.sum((module.output * module.grad), dim=(0, 2, 3))
                        values = values / (
                            module.output.size(0)
                            * module.output.size(2)
                            * module.output.size(3)
                        )
                    if hasattr(module, "output_mask"):
                        values[module.output_mask == 0] = EXTREMELY_HIGH_VALUE

                    self.filter_ranks[name] += (
                        values.cpu() if torch.cuda.is_available() else values
                    )

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            if self.args.norm:  # L2-norm for global rescaling
                if self.args.method_type == "lrp":  # it doesn't need
                    v = self.filter_ranks[i]
                    v = v / torch.sum(v[v < EXTREMELY_HIGH_VALUE])
                elif (
                    self.args.method_type == "weight"
                ):  # weight & L1-norm (Li et al., ICLR 2017)
                    v = self.filter_ranks[i]
                    v = v / torch.sum(
                        v[v < EXTREMELY_HIGH_VALUE]
                    )  # torch.sum(v) = total number of dataset
                elif (
                    self.args.method_type == "taylor"
                ):  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                    v = torch.abs(self.filter_ranks[i])
                    v = v / torch.sqrt(
                        torch.sum(
                            v[v < EXTREMELY_HIGH_VALUE] * v[v < EXTREMELY_HIGH_VALUE]
                        )
                    )
                elif (
                    self.args.method_type == "grad"
                ):  # |grad| & L2-norm (Sun et al., ICML 2017)
                    v = torch.abs(self.filter_ranks[i])
                    v = v / torch.sqrt(
                        torch.sum(
                            v[v < EXTREMELY_HIGH_VALUE] * v[v < EXTREMELY_HIGH_VALUE]
                        )
                    )
            else:
                if self.args.method_type == "weight":  # weight
                    v = self.filter_ranks[i]
                elif self.args.method_type == "taylor":  # |grad*act|
                    v = torch.abs(self.filter_ranks[i])
                elif self.args.method_type == "grad":  # |grad|
                    v = torch.abs(self.filter_ranks[i])
                elif self.args.method_type == "lrp":
                    v = self.filter_ranks[i]
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

        # Data list 내에서 가장 작은 수를 num(=512개) 만큼 뽑아서 리스트에 저장
        filters_to_prune = nsmallest(num, data, itemgetter(2))
        # Make sure the filters are unique
        assert (
            len(set([x[:2] for x in filters_to_prune])) == num
        ), "The selected filters should be unique"
        return filters_to_prune
