'''
Modified based on torchvision.models.alexnet.
'''
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.alexnet import model_urls


__all__ = ['AlexNet','alexnet']

class AlexNet(models.AlexNet):

    def __init__(self, *args, **kwargs):
        super(AlexNet, self).__init__(*args, **kwargs)
        self._out_features = self.classfier[6].in_features
        del self.classifier

        self.nonfc_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.nonfc_classifier(x)
        x = x.view(-1,self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features        


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model
