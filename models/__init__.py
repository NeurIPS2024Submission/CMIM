def get_network(netName='resnet18', Nclass=100):
    if netName == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(Nclass=Nclass)
    elif netName == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(Nclass=Nclass)
    elif netName == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(Nclass=Nclass)
    elif netName == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(Nclass=Nclass)
    elif netName == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(Nclass=Nclass)
    elif netName == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(Nclass=Nclass)
    elif netName == 'vgg11_bn':
        from models.vgg import  vgg11_bn
        net = vgg11_bn(Nclass=Nclass)
    elif netName == 'vgg13_bn':
        from models.vgg import vgg13_bn
        net = vgg13_bn(Nclass=Nclass)
    elif netName == 'vgg16_bn':
        from models.vgg import vgg16_bn
        net = vgg16_bn(Nclass=Nclass)
    elif netName == 'vgg19_bn':
        from models.vgg import vgg19_bn
        net = vgg19_bn(Nclass=Nclass)
    elif netName == "wrn40_2":
        from models.wideresidual import wideresnet
        net = wideresnet(Nclass=Nclass, depth=40, widen_factor=2)
    elif netName == "wrn16_2":
        from models.wideresidual import wideresnet
        net = wideresnet(Nclass=Nclass, depth=16, widen_factor=2)
    elif netName == "wrn40_1":
        from models.wideresidual import wideresnet
        net = wideresnet(Nclass=Nclass, depth=40, widen_factor=1)
    elif netName == "mobilenet":
        from models.mobilenet import mobilenet
        net = mobilenet(alpha=0.5, class_num=Nclass)
    return net
