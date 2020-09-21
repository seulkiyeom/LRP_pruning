import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import modules.data as dataset
from modules.filterprune import FilterPrunner
import pandas as pd
import os
from collections import Counter
import utils.lrp_general6 as lrp_alex
from modules.resnet_kuangliu import ResNet18_kuangliu_c, ResNet50_kuangliu_c
import modules.flops_counter_mask as fcm
import modules.flop as flop
from modules.prune_layer import prune_conv_layer

class PruningFineTuner:
    def __init__(self, args, model):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        lrp_params_def1 = {
            'conv2d_ignorebias': True,
            'eltwise_eps': 1e-6,
            'linear_eps': 1e-6,
            'pooling_eps': 1e-6,
            'use_zbeta': True,
        }

        lrp_layer2method = {
            'nn.ReLU': lrp_alex.relu_wrapper_fct,
            'nn.BatchNorm2d': lrp_alex.relu_wrapper_fct,
            'nn.Conv2d': lrp_alex.conv2d_beta0_wrapper_fct,
            'nn.Linear': lrp_alex.linearlayer_eps_wrapper_fct,
            'nn.AdaptiveAvgPool2d': lrp_alex.adaptiveavgpool2d_wrapper_fct,
            'nn.MaxPool2d': lrp_alex.maxpool2d_wrapper_fct,
            'sum_stacked2': lrp_alex.eltwisesum_stacked2_eps_wrapper_fct,
        }

        self.args = args
        self.setup_dataloaders()
        self.model = model

        if self.args.prune:
            self.wrapper_model = {
                'resnet18': ResNet18_kuangliu_c(),
                'resnet50': ResNet50_kuangliu_c(),
            }[self.args.arch.lower()]

            self.wrapper_model.copyfromresnet(self.model,
                                       lrp_params=lrp_params_def1,
                                       lrp_layer2method=lrp_layer2method)  # for test

            if self.args.method_type == 'lrp' or self.args.method_type == 'weight':
                self.prunner = FilterPrunner(self.wrapper_model, args)
            else:
                self.prunner = FilterPrunner(self.model, args)

        self.criterion = nn.CrossEntropyLoss() # log_softmax + NLL loss

        self.COUNT_ROW = 0
        self.COUNT_TRAIN = 0
        self.best_acc = 0
        self.save_loss = False

    def setup_dataloaders(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}

        # Data Acquisition
        get_dataset = {
            "cifar10": dataset.get_cifar10,  # CIFAR-10
            'imagenet': dataset.get_imagenet, # ImageNet
        }[self.args.data_type.lower()]
        train_dataset, test_dataset = get_dataset()
        print(f"train_dataset:{len(train_dataset)}, test_dataset:{len(test_dataset)}")
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.args.train_batch_size,
                                                        shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.args.test_batch_size,
                                                       shuffle=False, **kwargs)

        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)

    def total_num_filters(self):
        # count total number of filter in every conv layer
        filters = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "output_mask"):
                    filters += module.output_mask.sum()
                else:
                    filters += module.out_channels
        return filters

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            # optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        for i in range(epoches):
            print("Epoch: ", i)
            try: #during training
                scheduler.step()
                print(f"LR: {scheduler.get_lr()}")
                self.train_epoch(optimizer=optimizer)
                acc, _ = self.test()

                if acc > self.best_acc:
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    print(f"save a model")
                    save_loc = f"./checkpoint/{self.args.arch}_{self.args.data_type}_ckpt.pth"
                    torch.save(self.model.state_dict(), save_loc)

            except: #during fine-tuning
                self.train_epoch(optimizer=optimizer)
                test_accuracy, test_loss, flop_value, param_value = self.test()
                self.df.loc[self.COUNT_ROW] = pd.Series({"ratio_pruned": self.ratio_pruned_filters,
                                                    "test_acc": test_accuracy,
                                                    "test_loss": test_loss,
                                                    "flops": flop_value,
                                                    "params": param_value})
                self.COUNT_ROW += 1

        print("Finished fine tuning")


    def train_epoch(self, optimizer=None, rank_filters=False):
        self.train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.train_batch(optimizer, batch_idx, data, target, rank_filters)

            break

        if self.save_loss == True and rank_filters == False: #save train_loss only during fine-tuning
            self.dt.loc[self.COUNT_TRAIN] = pd.Series({"ratio_pruned": self.ratio_pruned_filters,
                                                       "train_loss": self.train_loss / len(self.train_loader.dataset)})
            self.COUNT_TRAIN += 1

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters):
        self.model.train()
        self.model.zero_grad()
        if optimizer is not None:
            optimizer.zero_grad()

        with torch.enable_grad():
            output = self.model(batch)

        if rank_filters: #for pruning
            batch.requires_grad = True
            if self.args.method_type == 'lrp' or self.args.method_type == 'weight':  # lrp_based
                with torch.enable_grad():
                    output = self.wrapper_model(batch)

                print("Computing LRP")

                # Map the original targets to [0, num_selected_classes)
                T = torch.zeros_like(output)
                for ii in range(len(label)):
                    T[ii, label[ii]] = 1.0

                # Multiply output with target
                lrp_anchor = output * T / (output * T).sum(dim=1, keepdim=True)
                output.backward(lrp_anchor, retain_graph=True)
                #
                input_relevance = batch.grad

                self.prunner.compute_filter_criterion(self.layer_type, criterion=self.args.method_type)

                print('Train Epoch: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(batch), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader)))
                print(
                    f"Sum of output {output.sum()}, input relevance {input_relevance.sum()}")
                self.train_loss += self.criterion(output, label).item()
                #
                # #Save heatmap
                # if batch_idx < 10:
                #     filename = Path(
                #         f"heatmaps/{args.arch}_trial{args.trialnum:02d}_batch{batch_idx:04d}_iter{epoch_idx:04d}.png")
                #     if not filename.parent.exists():
                #         filename.parent.mkdir()
                #     torchvision.utils.save_image(input_relevance, filename,
                #                                  normalize=True,
                #                                  scale_each=True)
            else:
                with torch.enable_grad():
                    output = self.prunner.model(batch)

                loss = self.criterion(output, label)
                loss.backward()
                self.train_loss += loss.item()

                input_relevance = batch.grad

                self.prunner.compute_filter_criterion(self.layer_type, criterion=self.args.method_type)

                print('Train Epoch: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(batch), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader)))
                print(
                    f"Sum of output {output.sum()}, input relevance {input_relevance.sum()}")

        else: #for normal training and fine-tuning
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.step()
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))
            self.train_loss += loss.item()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        ctr = 0

        for batch_idx, (data, target) in enumerate(self.test_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.model.zero_grad()
            with torch.enable_grad():
                output = self.model(data)

            test_loss += self.criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # print(f"True Label: {target},output: {output}")
            ctr += len(pred)

        test_loss /= ctr
        test_accuracy = float(correct) / ctr
        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{ctr} ({100 * test_accuracy:.5f}%)\n')
        # self.correct += correct

        if self.save_loss:
            sample_batch = torch.FloatTensor(1, 3, 32,
                                             32).cuda() if self.args.cuda else torch.FloatTensor(
                1, 3, 32, 32)
            self.model = fcm.add_flops_counting_methods(self.model)
            self.model.eval().start_flops_count()
            _ = self.model(sample_batch)
            flop_value = flop.flops_to_string_value(self.model.compute_average_flops_cost())
            param_value = flop.get_model_parameters_number_value_mask(self.model)
            self.model.eval().stop_flops_count()
            self.model = fcm.remove_flops_counting_methods(self.model)
            # print('Flops:', self.flop_val[-1])
            # print('Params:', self.num_param[-1])
            print(f'Flops: {flop_value}, Params: {param_value}')

            return test_accuracy, test_loss, flop_value, param_value

        else:
            return test_accuracy, test_loss

    def get_candidates_to_prune(self, num_filters_to_prune, layer_type='conv'):
        self.prunner.reset()
        self.layer_type = layer_type

        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()  # Normalization

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def copy_mask(self):
        my_model = self.model.modules()
        wrapper_model = self.wrapper_model.modules()

        for j in range(len(list(self.wrapper_model.modules()))):
            wrapper_module = next(wrapper_model)
            if hasattr(wrapper_module, "module") and isinstance(wrapper_module.module, nn.Conv2d):
                # print(f"wrapper: {wrapper_module.module}")
                for i in range(len(list(self.model.modules()))):
                    my_module = next(my_model)
                    if isinstance(my_module, nn.Conv2d):
                        if hasattr(my_module, "output_mask"):
                            wrapper_module.output_mask = my_module.output_mask
                            # print(f"module: {my_module}")
                        break
                    else:
                        continue
            else:
                continue

    def prune(self):
        self.save_loss = True
        self.model.eval()

        # Get the accuracy before pruning
        self.temp = 0
        test_accuracy, test_loss, flop_value, param_value = self.test()

        # Make sure all the layers are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(number_of_filters * self.args.pr_step)
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr)

        self.ratio_pruned_filters = 1.0
        results_file = f"scenario1_results_{self.args.data_type}_{self.args.arch}_{self.args.method_type}_trial{self.args.trialnum:02d}.csv"
        results_file_train = f"scenario1_train_{self.args.data_type}_{self.args.arch}_{self.args.method_type}_trial{self.args.trialnum:02d}.csv"
        self.df = pd.DataFrame(columns=["ratio_pruned", "test_acc", "test_loss", "flops","params"])
        self.dt = pd.DataFrame(columns=["ratio_pruned", "train_loss"])
        self.df.loc[self.COUNT_ROW] = pd.Series({"ratio_pruned": self.ratio_pruned_filters,
                                                 "test_acc": test_accuracy,
                                                 "test_loss": test_loss, "flops": flop_value,
                                                 "params": param_value})
        self.COUNT_ROW += 1
        for kk in range(iterations):
            print("Ranking filters.. {}".format(kk))
            prune_targets = self.get_candidates_to_prune(
                num_filters_to_prune_per_iteration, layer_type='conv')
            assert len(prune_targets) == num_filters_to_prune_per_iteration
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] += 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()  # 현재 모델 갖다가..
            ctr = self.total_num_filters()
            # Make sure that there are no duplicate filters
            assert len(set(prune_targets)) == len(prune_targets), [x for x in
                                                                   Counter(
                                                                       prune_targets).items()
                                                                   if x[1] > 1]
            for layer_index, filter_index in prune_targets:  # 하나씩 꺼내서 자르기 시작
                model = prune_conv_layer(model, layer_index, filter_index, criterion=self.args.method_type,
                                         cuda_flag=self.args.cuda)

                # Assert that one filter is pruned in each step
                ctr -= 1
                assert ctr == self.total_num_filters()

            self.model = model.cuda() if self.args.cuda else model
            assert self.total_num_filters() == number_of_filters - (
                    kk + 1) * num_filters_to_prune_per_iteration, self.total_num_filters()

            ratio_pruned_filters = float(
                self.total_num_filters()) / number_of_filters
            message = str(
                100 * ratio_pruned_filters) + "%"
            print("Filters prunned", str(message))
            test_accuracy, test_loss, flop_value, param_value = self.test()  # 잘리고 나서 test 해봄

            self.ratio_pruned_filters = ratio_pruned_filters
            self.df.loc[self.COUNT_ROW] = pd.Series({"ratio_pruned": ratio_pruned_filters,
                                                     "test_acc": test_accuracy,
                                                     "test_loss": test_loss,
                                                     "flops": flop_value,
                                                     "params": param_value})
            self.COUNT_ROW += 1
            self.df.to_csv(results_file)

            if self.args.method_type == 'lrp' or self.args.method_type == 'weight':
                self.copy_mask()  # copy mask from my model to the wrapper model

            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,
                                  momentum=self.args.momentum)
            self.train(optimizer, epoches=10)
            self.dt.to_csv(results_file_train)

        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=10)
        self.dt.to_csv(results_file_train)

        self.ratio_pruned_filters = ratio_pruned_filters
        self.df.loc[self.COUNT_ROW] = pd.Series({"ratio_pruned": ratio_pruned_filters,
                                                 "test_acc": test_accuracy,
                                                 "test_loss": test_loss,
                                                 "flops": flop_value,
                                                 "params": param_value})
        self.df.to_csv(results_file)

