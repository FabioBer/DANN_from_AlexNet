import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from DANN_from_AlexNet.functions import ReverseLayerF


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=7, num_domains=2):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.class_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
                 
        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_domains)
        )
        
    def update_weights(self):
        self.domain_classifier[1].weight.data = self.class_classifier[1].weight.data
        self.domain_classifier[1].bias.data = self.class_classifier[1].bias.data
        self.domain_classifier[4].weight.data = self.class_classifier[4].weight.data
        self.domain_classifier[4].bias.data = self.class_classifier[4].bias.data
        
    def forward(self, input_data, alpha=None):
        feature = self.features(input_data)
        feature = self.avgpool(feature)
        feature = torch.flatten(feature, 1)
        if alpha is None:
            return self.class_classifier(feature)
        else:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            return self.domain_classifier(feature)
    
    
def DANN_from_AlexNet(pretrained=False, progress=True, **kwargs):
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
         
        # unused parameters removal
        state_dict.popitem("classifier.6.bias")
        state_dict.popitem("classifier.6.weight")
        
        model.load_state_dict(state_dict,strict=False)
        model.update_weights()

    return model
