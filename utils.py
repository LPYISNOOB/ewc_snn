from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import spikingjelly as sj
from spikingjelly.activation_based.encoding import PoissonEncoder
encoder = PoissonEncoder()
def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


# def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
#     model.train()
#     epoch_loss = 0
#     for input, target in data_loader:
#         input, target = variable(input), variable(target)
#         optimizer.zero_grad()
#         output = model(input)
#         loss = F.cross_entropy(output, target)
#         # epoch_loss += loss.data[0]
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     return epoch_loss / len(data_loader)


# def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
#               ewc: EWC, importance: float):
#     model.train()
#     epoch_loss = 0
#     for input, target in data_loader:
#         input, target = variable(input), variable(target)
#         optimizer.zero_grad()
#         output = model(input)
#         loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     return epoch_loss / len(data_loader)


# def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
#     model.eval()
#     correct = 0
#     for input, target in data_loader:
#         input, target = variable(input), variable(target)
#         output = model(input)
#         correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
#     return correct / len(data_loader.dataset)


# 假设你的SNN模型已经被定义，并且有一个名为encoder的PoissonEncoder实例

def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, encoder, T=100):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        # input = input.to(model.device)  # 确保输入数据在正确的设备上
        # target = target.to(model.device)  # 确保目标数据在正确的设备上
        optimizer.zero_grad()
        
        # 在多个时间步长上累积输出
        output = 0
        for t in range(T):
            encoded_input = encoder(input)
            output += model(encoded_input)
            functional.reset_net(model)
        output = output / T  # 计算平均输出
        loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float())  # 使用均方误差损失
        epoch_loss += loss.item()
        
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        functional.reset_net(model)  # 重置网络状态
        
    return epoch_loss / len(data_loader)

def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float, encoder, T=100):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input = input.to(model.device)  # 确保输入数据在正确的设备上
        target = target.to(model.device)  # 确保目标数据在正确的设备上
        optimizer.zero_grad()
        
        # 在多个时间步长上累积输出
        output = 0
        for t in range(T):
            encoded_input = encoder(input)
            output += model(encoded_input)
            functional.reset_net(model)
        output = output / T  # 计算平均输出
        loss = F.mse_loss(output, F.one_hot(target, num_classes=10).float())  # 使用均方误差损失
        loss += importance * ewc.penalty(model)  # 添加EWC惩罚项
        epoch_loss += loss.item()
        
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        functional.reset_net(model)  # 重置网络状态
        
    return epoch_loss / len(data_loader)

def test(model: nn.Module, data_loader: torch.utils.data.DataLoader, encoder, T=100):
    model.eval()
    correct = 0
    with torch.no_grad():  # 在评估模式下，不计算梯度
        for input, target in data_loader:
            # input = input.to(model.device)  # 确保输入数据在正确的设备上
            # target = target.to(model.device)  # 确保目标数据在正确的设备上
            
            # 在多个时间步长上累积输出
            output = 0
            for t in range(T):
                encoded_input = encoder(input)
                output += model(encoded_input)
                functional.reset_net(model)
            output = output / T  # 计算平均输出
            _, predicted = torch.max(output, 1)  # 获取预测结果
            correct += (predicted == target).sum().item()
            
    return correct / len(data_loader.dataset)  # 返回准确率