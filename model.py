import torch.nn as nn
from roi_pooling import roi_pooling as _roi_pooling

from rpn import RPN as _RPN
from faster_rcnn import FasterRCNN as _FasterRCNN

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class _Features(nn.Container):
  def __init__(self):
    super(_Features, self).__init__()
    self.m = make_layers(cfgs['E'], batch_norm=True)
    self._initialize_weights()

  def forward(self, x):
    return self.m(x)

  def _initialize_weights(self):
    for m in self.m.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, 0, 0.01)
          nn.init.constant_(m.bias, 0)

class _Classifier(nn.Container):
  def __init__(self):
    super(_Classifier, self).__init__()
    self.m1 = nn.Linear(512*7*7, 13)
    self.m2 = nn.Linear(512*7*7, 13*4)
    self.m3 = nn.Linear(512*7*7, 13*24)

  def forward(self, x):
    return self.m1(x), self.m2(x), self.m3(x)

def _pooler(x, rois):
  x = _roi_pooling(x, rois, size=(7,7), spatial_scale=1.0/16.0)
  return x.view(x.size(0), -1)

class _RPNClassifier(nn.Container):
  def __init__(self, n):
    super(_RPNClassifier, self).__init__()
    self.m1 = nn.Conv2d(n, 18, 3, 1, 1)
    self.m2 = nn.Conv2d(n, 36, 3, 1, 1)

  def forward(self, x):
    return self.m1(x), self.m2(x)

def model():
  _features = _Features()
  _classifier = _Classifier()
  _rpn_classifier = _RPNClassifier(512)

  _rpn = _RPN(
    classifier=_rpn_classifier
  )
  _model = _FasterRCNN(
    features=_features,
    pooler=_pooler,
    classifier=_classifier,
    rpn=_rpn
  )
  return _model
