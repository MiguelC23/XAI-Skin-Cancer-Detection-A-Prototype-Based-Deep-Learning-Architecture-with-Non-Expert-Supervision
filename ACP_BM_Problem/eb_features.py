import torch
import torch.nn as nn
import torchvision.models as models

def efficientnet_b0_features(pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b1_features(pretrained=True):
    model = models.efficientnet_b1(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b2_features(pretrained=True):
    model = models.efficientnet_b2(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b3_features(pretrained=True):
    model = models.efficientnet_b3(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b4_features(pretrained=True):
    model = models.efficientnet_b4(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b5_features(pretrained=True):
    model = models.efficientnet_b5(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b6_features(pretrained=True):
    model = models.efficientnet_b6(pretrained=pretrained)
    features = model.features
    return features

def efficientnet_b7_features(pretrained=True):
    model = models.efficientnet_b7(pretrained=pretrained)
    features = model.features
    return features


if __name__ == '__main__':

    eb0_features=efficientnet_b0_features(pretrained=True)
    print(eb0_features)

    eb1_features=efficientnet_b1_features(pretrained=True)
    print(eb1_features)

    eb2_features=efficientnet_b2_features(pretrained=True)
    print(eb2_features)

    eb3_features=efficientnet_b3_features(pretrained=True)
    print(eb3_features)

    eb4_features=efficientnet_b4_features(pretrained=True)
    print(eb4_features)

    eb5_features=efficientnet_b5_features(pretrained=True)
    print(eb5_features)

    eb6_features=efficientnet_b6_features(pretrained=True)
    print(eb6_features)

    eb7_features=efficientnet_b7_features(pretrained=True)
    print(eb7_features)