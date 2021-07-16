from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import lse

affine_par = True


class DelseModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.T = opts.delse_iterations
        self.dt_max = opts.dt_max
        self.input_channels = opts.input_channels
        n_classes = (1, 2, 1)

        concat_dim = 128
        feature_dim = 4 * concat_dim
        layers = (3, 4, 23, 3)
        dilations = (2, 4)
        strides = (2, 2, 2, 1, 1)
        model = ResNet(Bottleneck, layers, n_classes[0],
                       nInputChannels=self.input_channels + 1, classifier="psp",
                       dilations=dilations, strides=strides, _print=True,
                       feature_dim=feature_dim)

        model_full = Res_Deeplab(opts.delse_pth, n_classes[0])
        model.load_pretrained_ms(model_full, nInputChannels=self.input_channels + 1)
        model.layer5_1 = PSPModule(in_features=feature_dim, out_features=512,
                                   sizes=(1, 2, 3, 6), n_classes=n_classes[1])
        model.layer5_2 = PSPModule(in_features=feature_dim, out_features=512,
                                   sizes=(1, 2, 3, 6), n_classes=n_classes[2])

        weight_init(model.layer5_1)
        weight_init(model.layer5_2)
        self.full_model = SkipResnet(concat_channels=concat_dim, resnet=model)

    def forward(self, x, meta):
        x = torch.cat([meta[:, 0:1], x], dim=1)  # add extreme points labels
        outputs = self.full_model(x)
        phi_0, energy, g = [lse.interpolater(z, x.shape[2:4]) for z in outputs]
        return [phi_0, energy, torch.sigmoid(g)]

    def infer(self, x, meta):
        with torch.no_grad():
            phi_0, energy, g = self.forward(x, meta)
            probs = lse.levelset_evolution(phi_0, energy, g, self.T, self.dt_max)
            return torch.argmax(probs, dim=1), probs


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i


class ClassifierModule(nn.Module):
    def __init__(self, dilation_series, padding_series, n_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(512, n_classes, kernel_size=3, stride=1,  # had been 2048, not 512
                          padding=padding, dilation=dilation, bias=True)
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """

    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_features, n_classes, kernel_size=1)

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        for i in bn.parameters():
            i.requires_grad = False

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        for i in bn.parameters():
            i.requires_grad = False

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3,
                 classifier="atrous", dilations=(2, 4), strides=(2, 2, 2, 1, 1),
                 _print=True, feature_dim=2048):
        if _print:
            print("Constructing ResNet model...")
            print("Dilations: {}".format(dilations))
            print("Number of classes: {}".format(n_classes))
            print("Number of Input Channels: {}".format(nInputChannels))
        self.inplanes = 64
        self.classifier = classifier
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])

        print('Initializing classifier: A-trous pyramid')
        self.layer5 = ClassifierModule([6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        self.layer5_1 = None
        self.layer5_2 = None
        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = [block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.layer5_2 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            x2 = self.layer5_2(x)
            return x0, x1, x2
        if self.layer5_1 is not None:
            x0 = self.layer5(x)
            x1 = self.layer5_1(x)
            return x0, x1
        if self.layer5 is not None:
            x0 = self.layer5(x)
            return x0
        return x

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for module, module_ori in zip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) \
                    and module.weight.data.shape == module_ori.weight.data.shape:
                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [self.layer5, self.layer5_1, self.layer5_2]
        for j in range(len(b)):
            if b[j] is not None:
                for k in b[j].parameters():
                    if k.requires_grad:
                        yield k


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, nInputChannels=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, nInputChannels=nInputChannels)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear', align_corners=True)
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear', align_corners=True)
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear', align_corners=True)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))  # for original scale
        out.append(self.interp3(self.Scale(x2)))  # for 0.75x scale
        out.append(self.Scale(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


def Res_Deeplab(pth_model, n_classes=21, pretrained=True):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained:
        saved_state_dict = torch.load(pth_model, map_location=lambda x, loc: x)
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
    return model


class SkipResnet(nn.Module):
    def __init__(self, concat_channels=128, mid_dim=256, final_dim=512, resnet=None, torch_model=False, use_conv=False):
        super(SkipResnet, self).__init__()

        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

        self.concat_channels = concat_channels
        self.final_dim = final_dim

        assert resnet is not None
        self.resnet = resnet

        self.torch_model = torch_model
        self.use_conv = use_conv

        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.res1_concat = nn.Sequential(concat2, bn2, relu2, up2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.res2_concat = nn.Sequential(concat3, bn3, relu3, up3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)

        # Different from original, original used maxpool
        # Original used no activation here
        if self.use_conv:
            conv_final_1 = nn.Conv2d(4*concat_channels, mid_dim, kernel_size=3,
                                     padding=1, stride=2, bias=False)
            bn_final_1 = nn.BatchNorm2d(mid_dim)
            conv_final_2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, stride=2, bias=False)
            bn_final_2 = nn.BatchNorm2d(mid_dim)
            conv_final_3 = nn.Conv2d(mid_dim, final_dim, kernel_size=3, padding=1, bias=False)
            bn_final_3 = nn.BatchNorm2d(final_dim)

            self.conv_final = nn.Sequential(conv_final_1, bn_final_1,
                                            conv_final_2, bn_final_2,
                                            conv_final_3, bn_final_3)
        else:
            self.conv_final = None

    def forward(self, x):
        if self.torch_model:
            x = self.normalize(x)
        # Normalization

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        conv1_f = self.resnet.relu(x)
        x = self.resnet.maxpool(conv1_f)
        layer1_f = self.resnet.layer1(x)
        layer2_f = self.resnet.layer2(layer1_f)
        layer3_f = self.resnet.layer3(layer2_f)
        layer4_f = self.resnet.layer4(layer3_f)

        conv1_f = self.conv1_concat(conv1_f)
        layer1_f = self.res1_concat(layer1_f)
        layer2_f = self.res2_concat(layer2_f)
        layer4_f = self.res4_concat(layer4_f)

        concat_features = torch.cat((conv1_f, layer1_f, layer2_f, layer4_f), dim=1)
        if self.use_conv:
            x = self.conv_final(concat_features)    # final feature map
        else:
            x = concat_features

        # classifiers
        if self.resnet.layer5_2 is not None:
            x0 = self.resnet.layer5(x)
            x1 = self.resnet.layer5_1(x)
            x2 = self.resnet.layer5_2(x)
            return x0, x1, x2
        if self.resnet.layer5_1 is not None:
            x0 = self.resnet.layer5(x)
            x1 = self.resnet.layer5_1(x)
            return x0, x1
        if self.resnet.layer5 is not None:
            x0 = self.resnet.layer5(x)
            return x0
        return x

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this
        function does not return any batchnorm parameter
        """
        b = [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1,
             self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the
        net, which does the classification of pixel into classes
        """
        b = [self.resnet.layer5, self.resnet.layer5_1, self.resnet.layer5_2,
             self.conv1_concat, self.res1_concat, self.res2_concat,
             self.res4_concat, self.conv_final]
        for j in range(len(b)):
            if b[j] is not None:
                for k in b[j].parameters():
                    if k.requires_grad:
                        yield k
