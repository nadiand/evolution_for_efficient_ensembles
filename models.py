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
import torchvision


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


def load_pascal_models():
    segmentors = []

    model1 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_12345.pth")
    model1.load_state_dict(torch.load(state_dict)['model'])
    model1.eval()
    segmentors.append(model1)

    model2 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_32823.pth")
    model2.load_state_dict(torch.load(state_dict)['model'])
    model2.eval()
    segmentors.append(model2)

    model3 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_57923.pth")
    model3.load_state_dict(torch.load(state_dict)['model'])
    model3.eval()
    segmentors.append(model3)

    model4 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_70852.pth")
    model4.load_state_dict(torch.load(state_dict)['model'])
    model4.eval()
    segmentors.append(model4)

    model5 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_97245.pth")
    model5.load_state_dict(torch.load(state_dict)['model'])
    model5.eval()
    segmentors.append(model5)

    model6 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_126181.pth")
    model6.load_state_dict(torch.load(state_dict)['model'])
    model6.eval()
    segmentors.append(model6)

    
    model7 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_10new.pth")
    model7.load_state_dict(torch.load(state_dict)['model'])
    model7.eval()
    segmentors.append(model7)

    model8 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_1000new.pth")
    model8.load_state_dict(torch.load(state_dict)['model'])
    model8.eval()
    segmentors.append(model8)

    model9 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_9000new.pth")
    model9.load_state_dict(torch.load(state_dict)['model'])
    model9.eval()
    segmentors.append(model9)

    model10 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_12000new.pth")
    model10.load_state_dict(torch.load(state_dict)['model'])
    model10.eval()
    segmentors.append(model10)

    model11 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_15000new.pth")
    model11.load_state_dict(torch.load(state_dict)['model'])
    model11.eval()
    segmentors.append(model11)

    model12 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_19000new.pth")
    model12.load_state_dict(torch.load(state_dict)['model'])
    model12.eval()
    segmentors.append(model12)

    model13 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_22000new.pth")
    model13.load_state_dict(torch.load(state_dict)['model'])
    model13.eval()
    segmentors.append(model13)

    model14 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_25000new.pth")
    model14.load_state_dict(torch.load(state_dict)['model'])
    model14.eval()
    segmentors.append(model14)

    model15 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_28000new.pth")
    model15.load_state_dict(torch.load(state_dict)['model'])
    model15.eval()
    segmentors.append(model15)
    
    return segmentors


def load_pascal_preds():
    segmentors = []
    for i in range(0, 15):
        preds = torch.load(f"/dataB3/nadia_dobreva/model{i}_tensor_preds.pt")
        segmentors.append(preds)
    return segmentors


def load_pascal_weighted_models():
    segmentors = []
    model4 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_70852.pth")
    model4.load_state_dict(torch.load(state_dict)['model'])
    model4.eval()
    segmentors.append([model4,0.9940257412252314])

    model = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_57923.pth")
    model.load_state_dict(torch.load(state_dict)['model'])
    model.eval()
    segmentors.append([model,1.133008297907566])

    model2 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_97245.pth")
    model2.load_state_dict(torch.load(state_dict)['model'])
    model2.eval()
    segmentors.append([model2,1.2221001980005783])

    model3 = torchvision.models.get_model(
        "deeplabv3_mobilenet_v3_large",
        weights=None,
        weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        num_classes=21,
        aux_loss=True,
    )
    state_dict = os.path.join("pascal_models", "state_dicts", "pascalvoc_mobilenetv3_126181.pth")
    model3.load_state_dict(torch.load(state_dict)['model'])
    model3.eval()
    segmentors.append([model3,1.0939830006870825])

    return segmentors
