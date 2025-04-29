import torch
import torch.nn as nn
from torch import Tensor
from transfer_losses import TransferLoss
import backbones
from torchsummaryX import summary

def EM(prediction):
    p = nn.Softmax(prediction,dim=-1)
    return -1 * torch.sum(p * nn.LogSoftmax(prediction, dim=-1)) / prediction.size()[0]

def EM(prediction):
    # Apply softmax to get probabilities
    p = torch.softmax(prediction, dim=-1)  # Convert logits to probabilities

    # Compute log probabilities
    log_p = torch.log(p)

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(p * log_p, dim=-1).mean()  # Mean over batch

    return entropy

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        '''self.classifier_layer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(feature_dim, num_class),
        )'''
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    '''def forward(self, x: Tensor) -> Tensor:
        x = self.base_network(x)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        x = self.classifier_layer(x)
        return x'''

    def forward(self, source, target, source_label, target_label):
        source_feature = self.base_network(source)
        target_feature = self.base_network(target)
        if self.use_bottleneck:
            source_feature = self.bottleneck_layer(source_feature)
            target_feature = self.bottleneck_layer(target_feature)
        # classification
        source_clf = self.classifier_layer(source_feature)
        target_clf = self.classifier_layer(target_feature)
        source_clf_loss = self.criterion(source_clf, source_label)
        target_clf_loss = self.criterion(target_clf, target_label)
        # transfer
        kwargs = {}
        source_clf = self.classifier_layer(source_feature)
        kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
        target_clf = self.classifier_layer(target_feature)
        kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)

        transfer_loss = self.adapt_loss(source_feature, target_feature, **kwargs)
        return transfer_loss, source_clf_loss, target_clf_loss
        # return transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters()},
            {'params': self.classifier_layer.parameters()},
        ]
        if self.use_bottleneck:
            params.append(
                #{'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
                {'params': self.bottleneck_layer.parameters()}
            )
        # Loss-dependent
        params.append(
            {'params': self.adapt_loss.loss_func.domain_classifier.parameters()}
        )
        params.append(
            {'params': self.adapt_loss.loss_func.local_classifiers.parameters()}
        )
        return params

    def predict(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass