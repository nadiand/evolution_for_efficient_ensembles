from cifar100_models.resnet import CifarResNet, BasicBlock
from cifar100_models.mobilenetv2 import MobileNetV2
from cifar100_models.shufflenetv2 import ShuffleNetV2
from cifar100_models.vgg import VGG
from cifar10_models.densenet import densenet121, densenet169, densenet161
from cifar10_models.resnet import resnet18, resnet34
from cifar10_models.googlenet import googlenet
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn
import os
import torch


def load_cifar10_models():
    classifiers = []

    densenet_model = densenet121()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet121" + ".pt")
    densenet_model.load_state_dict(torch.load(state_dict))
    densenet_model.eval()
    classifiers.append(densenet_model)

    densenet_model2 = densenet169()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet169" + ".pt")
    densenet_model2.load_state_dict(torch.load(state_dict))
    densenet_model2.eval()
    classifiers.append(densenet_model2)

    resnet_model = resnet18()
    state_dict = os.path.join("cifar10_models", "state_dicts", "resnet18" + ".pt")
    resnet_model.load_state_dict(torch.load(state_dict))
    resnet_model.eval()
    classifiers.append(resnet_model)

    resnet_model2 = resnet34()
    state_dict = os.path.join("cifar10_models", "state_dicts", "resnet34" + ".pt")
    resnet_model2.load_state_dict(torch.load(state_dict))
    resnet_model2.eval()
    classifiers.append(resnet_model2)

    googlenet_model = googlenet()
    state_dict = os.path.join("cifar10_models", "state_dicts", "googlenet" + ".pt")
    googlenet_model.load_state_dict(torch.load(state_dict))
    googlenet_model.eval()
    classifiers.append(googlenet_model)

    mobilenet_v2_model = mobilenet_v2()
    state_dict = os.path.join("cifar10_models", "state_dicts", "mobilenet_v2" + ".pt")
    mobilenet_v2_model.load_state_dict(torch.load(state_dict))
    mobilenet_v2_model.eval()
    classifiers.append(mobilenet_v2_model)

    densenet_model3 = densenet161()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet161" + ".pt")
    densenet_model3.load_state_dict(torch.load(state_dict))
    densenet_model3.eval()
    classifiers.append(densenet_model3)

    vgg_model = vgg11_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg11_bn" + ".pt")
    vgg_model.load_state_dict(torch.load(state_dict))
    vgg_model.eval()
    classifiers.append(vgg_model)

    vgg_model2 = vgg13_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg13_bn" + ".pt")
    vgg_model2.load_state_dict(torch.load(state_dict))
    vgg_model2.eval()
    classifiers.append(vgg_model2)

    vgg_model3 = vgg16_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg16_bn" + ".pt")
    vgg_model3.load_state_dict(torch.load(state_dict))
    vgg_model3.eval()
    classifiers.append(vgg_model3)

    print('models loaded')
    return classifiers


def load_cifar100_models():
    classifiers = []

    resnet_model = CifarResNet(BasicBlock, [3]*3, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_resnet20" + ".pt")
    resnet_model.load_state_dict(torch.load(state_dict))
    resnet_model.eval()
    classifiers.append(resnet_model)

    resnet_model2 = CifarResNet(BasicBlock, [5]*3, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_resnet32" + ".pt")
    resnet_model2.load_state_dict(torch.load(state_dict))
    resnet_model2.eval()
    classifiers.append(resnet_model2)

    resnet_model3 = CifarResNet(BasicBlock, [7]*3, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_resnet44" + ".pt")
    resnet_model3.load_state_dict(torch.load(state_dict))
    resnet_model3.eval()
    classifiers.append(resnet_model3)

    resnet_model4 = CifarResNet(BasicBlock, [9]*3, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_resnet56" + ".pt")
    resnet_model4.load_state_dict(torch.load(state_dict))
    resnet_model4.eval()
    classifiers.append(resnet_model4)

    mobilenet_model = MobileNetV2(width_mult=0.5, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_mobilenetv2_x0_5" + ".pt")
    mobilenet_model.load_state_dict(torch.load(state_dict))
    mobilenet_model.eval()
    classifiers.append(mobilenet_model)

    mobilenet_model2 = MobileNetV2(width_mult=0.75, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_mobilenetv2_x0_75" + ".pt")
    mobilenet_model2.load_state_dict(torch.load(state_dict))
    mobilenet_model2.eval()
    classifiers.append(mobilenet_model2)

    mobilenet_model3 = MobileNetV2(width_mult=1.0, num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_mobilenetv2_x1_0" + ".pt")
    mobilenet_model3.load_state_dict(torch.load(state_dict))
    mobilenet_model3.eval()
    classifiers.append(mobilenet_model3)

    shuffle_model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_shufflenetv2_x1_0.pt")
    shuffle_model.load_state_dict(torch.load(state_dict))
    shuffle_model.eval()
    classifiers.append(shuffle_model)

    shuffle_model2 = ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_shufflenetv2_x1_5.pt")
    shuffle_model2.load_state_dict(torch.load(state_dict))
    shuffle_model2.eval()
    classifiers.append(shuffle_model2)

    shuffle_model3 = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=100)
    state_dict = os.path.join("cifar100_models", "state_dict", "cifar100_shufflenetv2_x0_5.pt")
    shuffle_model3.load_state_dict(torch.load(state_dict))
    shuffle_model3.eval()
    classifiers.append(shuffle_model3)

    print('models loaded')
    return classifiers