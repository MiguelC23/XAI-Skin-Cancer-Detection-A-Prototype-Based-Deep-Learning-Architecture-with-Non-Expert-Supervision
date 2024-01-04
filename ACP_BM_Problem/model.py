import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2
from eb_features import efficientnet_b0_features,efficientnet_b1_features,efficientnet_b2_features,efficientnet_b3_features,\
                                    efficientnet_b4_features,efficientnet_b5_features,efficientnet_b6_features,efficientnet_b7_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features,
                                 'eb0' : efficientnet_b0_features,
                                 'eb1' : efficientnet_b1_features,
                                 'eb2' : efficientnet_b2_features,
                                 'eb3' : efficientnet_b3_features,
                                 'eb4' : efficientnet_b4_features,
                                 'eb5' : efficientnet_b5_features,
                                 'eb6' : efficientnet_b6_features,
                                 'eb7' : efficientnet_b7_features}


class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, topk_k=1, num_classes=3, init_weights=True, last_layer_connection_weight=None,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 LP_MASKED=False,Base_architecture=None,Fixed_prototypes_during_training_initialized_orthogonally=False):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.topk_k = topk_k 
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.last_layer_connection_weight = last_layer_connection_weight
        self.LP_MASKED=LP_MASKED
        self.base_architecture=Base_architecture
        self.Fixed_prototypes_during_training_initialized_orthogonally=Fixed_prototypes_during_training_initialized_orthogonally
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features
        features_name = str(self.features).upper()

        #from settings import base_architecture
        if features_name.startswith('VGG') or features_name.startswith('RES') or features_name.startswith('DENSE'):
            print("Not eff net\n")
        elif self.base_architecture.startswith('eb'):
            features_name='EFFICIENTNET'
            print(self.base_architecture)
            print("Eff net\n")


        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        elif features_name.startswith('EFFICIENTNET'):
            last_conv_layer = features[-1]
            num_output_features = last_conv_layer.out_channels
            first_add_on_layer_in_channels=num_output_features
            print(first_add_on_layer_in_channels)
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        if(self.Fixed_prototypes_during_training_initialized_orthogonally==False):
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),requires_grad=True)
        elif(self.Fixed_prototypes_during_training_initialized_orthogonally==True):
            self.prototype_vectors = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(*self.prototype_shape)),requires_grad=False)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()


    
    
    def set_topk_k(self, topk_k):
        '''set the topk_k'''
        self.topk_k = topk_k
    
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        batch = x.shape[0]
        # x is the conv output, shape=[Batch * channel * conv output shape]
        expanded_x = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(x)
        #print('1expanded_x.shape:',expanded_x.shape)
        expanded_x = expanded_x.unsqueeze(0).permute(0,1,3,2)
        #print('2expanded_x.shape:',expanded_x.shape)
        # expanded shape = [1, batch, number of such blocks, channel*proto_shape[2]*proto_shape[3]]
        expanded_x = expanded_x.contiguous().view(1, -1, self.prototype_shape[1] *self.prototype_shape[2] * self.prototype_shape[3])
        #print('3expanded_x.shape:',expanded_x.shape)
        expanded_proto = nn.Unfold(kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(self.prototype_vectors).unsqueeze(0)
        #print('expanded_proto.shape:',expanded_proto.shape)
        # expanded proto shape = [1, proto num, channel*proto_shape[2]*proto_shape[3], 1]
        #print( expanded_proto.contiguous().view(1, expanded_proto.shape[1], -1).shape)
        expanded_distances = torch.cdist(expanded_x, expanded_proto.contiguous().view(1, expanded_proto.shape[1], -1))
        #print('1expanded_distances.shape:',expanded_distances.shape)
        # [1, Batch * number of blocks in x, num proto]
        expanded_distances = torch.reshape(expanded_distances, shape=(batch, -1, self.prototype_shape[0])).permute(0,2,1)
        #print('2expanded_distances.shape:',expanded_distances.shape)
        # print(expanded_distances.shape)
        # distances = nn.Fold(output_size=(x.shape[2] - self.prototype_shape[2] + 1, x.shape[3]- self.prototype_shape[3] + 1), kernel_size=(self.prototype_shape[2], self.prototype_shape[3]))(expanded_distances)
        distances = torch.reshape(expanded_distances, shape=(batch, self.prototype_shape[0], x.shape[2] - self.prototype_shape[2] + 1, x.shape[3] - self.prototype_shape[3] + 1))
        # distance shape = [batch, proto num, conv output shape]
        #print('distances.shape:',distances.shape)
        
        return distances
    
    
    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances
    

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, mascaras):
        # x is of dimension (batch, 4, spatial, spatial)
        x = x[:, 0:3, :, :]  #(no view; create slice. When no fa is available this will return x)~
        #print(x.shape)
        distances = self.prototype_distances(x)
        #print(distances.shape) #[batch_size,numP,7,7] or #[batch_size,numP,14,14]
        if (self.LP_MASKED==True):
            if(mascaras!=None):
                mascaras_reduzidas=F.adaptive_avg_pool2d(mascaras, (distances.shape[2], distances.shape[3]))
                ones_tensor = torch.ones(mascaras_reduzidas.shape[0], mascaras_reduzidas.shape[1], mascaras_reduzidas.shape[2], mascaras_reduzidas.shape[3]).cuda()
                max_dist = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
                passo_intermedio=mascaras_reduzidas * max_dist
                distances= distances * (ones_tensor - mascaras_reduzidas) + passo_intermedio

        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''    
        _distances = distances.view(distances.shape[0], distances.shape[1], -1)
        top_k_neg_distances, _ = torch.topk(-_distances, self.topk_k)
        closest_k_distances = - top_k_neg_distances
        min_distances = F.avg_pool1d(closest_k_distances, kernel_size=closest_k_distances.shape[2]).view(-1, self.num_prototypes)

        prototype_activations = self.distance_2_similarity(distances)
        _activations = prototype_activations.view(prototype_activations.shape[0], prototype_activations.shape[1], -1)
        top_k_activations, _ = torch.topk(_activations, self.topk_k)
        prototype_activations = F.avg_pool1d(top_k_activations, kernel_size=top_k_activations.shape[2]).view(-1, self.num_prototypes) 
        
        logits = self.last_layer(prototype_activations)
        
        activation = self.distance_2_similarity(distances)
        # UPSAMPLE IS NOT DETERMINISTIC
        #upsampled_activation = torch.nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)(activation) 
        upsampled_activation = F.adaptive_avg_pool2d(activation, (x.shape[2], x.shape[3]))       
        return logits, min_distances, upsampled_activation

    def push_forward(self, x, mascaras):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        if (self.LP_MASKED==True):
            if(mascaras!=None):
                mascaras_reduzidas=F.adaptive_avg_pool2d(mascaras, (distances.shape[2], distances.shape[3]))
                ones_tensor = torch.ones(mascaras_reduzidas.shape[0], mascaras_reduzidas.shape[1], mascaras_reduzidas.shape[2], mascaras_reduzidas.shape[3]).cuda()
                max_dist = self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3]
                passo_intermedio=mascaras_reduzidas * max_dist
                distances= distances * (ones_tensor - mascaras_reduzidas) + passo_intermedio

        
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tnum_prototypes: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {},\n'
            '\tbase_architecture: {},\n'
            '\tLP_MASKED: {},\n'
            '\tFixed_prototypes_during_training_initialized_orthogonally: {},\n'
            '\ttopk_k: {},\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.num_prototypes,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon,
                          self.base_architecture,
                          self.LP_MASKED,
                          self.Fixed_prototypes_during_training_initialized_orthogonally,
                          self.topk_k)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.last_layer_connection_weight:
            self.set_last_layer_incorrect_connection(incorrect_strength=self.last_layer_connection_weight)
        else:
            self.set_last_layer_incorrect_connection(incorrect_strength=-2)

def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), topk_k=1, num_classes=200,
                    prototype_activation_function='log', last_layer_weight=None,
                    add_on_layers_type='bottleneck',LP_MASKED=False,Fixed_prototypes_during_training_initialized_orthogonally=False):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    if(base_architecture.startswith('eb')):
        kernel_sizes = []
        strides = []
        paddings = []        
        for module in features.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
                kernel_sizes.append(module.kernel_size[0])
                strides.append(module.stride[0])
                paddings.append(module.padding[0])

        layer_filter_sizes=kernel_sizes
        layer_strides=strides
        layer_paddings=paddings
    else:
        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()


    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 topk_k=topk_k,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 last_layer_connection_weight=last_layer_weight,
                 add_on_layers_type=add_on_layers_type,LP_MASKED=LP_MASKED,Base_architecture=base_architecture,Fixed_prototypes_during_training_initialized_orthogonally=Fixed_prototypes_during_training_initialized_orthogonally)

