import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    wandb=print,
                    diversity=False):

    prototype_network_parallel.eval()
    log('\tpush')
    

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y, ISIC_ID,mask) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        if search_batch_input.shape[1] == 4:
            search_batch_input = search_batch_input[:, 0:3, :, :]  #(no view, create slice)

        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   patient_id= None,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                                   diversity=diversity,
                                   ISIC_ID=ISIC_ID,
                                   mask=mask)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    if(prototype_network_parallel.module.Fixed_prototypes_during_training_initialized_orthogonally==False): # If prototypes are not fixed during training update them 
        prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
        prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               patient_id=None,
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               diversity=False,
                               ISIC_ID=None,
                               mask=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch,mask.cuda())

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    # if class_specific:
    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        img_label = img_y.item()
        class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    
    if(diversity==True):
        idxs_given_from_batch_to_proto= np.full(n_prototypes,-2)

    for j in range(prototype_network_parallel.module.num_malignant_prototypes):# Only Melanoma Prototypes
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            # proto_dist_j = proto_dist_[:,j,:,:]
            target_class = 1
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]

        if(diversity==True):
            args_ordered_proto_dist_j=np.argsort(proto_dist_j,axis=None)
            proto_dist_j_flatten=proto_dist_j.flatten()
            for idx in args_ordered_proto_dist_j:
                possible_min_value=proto_dist_j_flatten[idx]
                idx_imagem=class_to_img_index_dict[target_class][list(np.unravel_index(idx,proto_dist_j.shape))[0]]
                condition= (possible_min_value<global_min_proto_dist[j] and (idx_imagem not in idxs_given_from_batch_to_proto))
                if(condition==True):
                    batch_min_proto_dist_j=possible_min_value
                    idxs_given_from_batch_to_proto[j]=idx_imagem
                    idx_yes_diversity=idx
                    break

        if(diversity==False):
            batch_min_proto_dist_j = np.amin(proto_dist_j)
            condition=(batch_min_proto_dist_j < global_min_proto_dist[j])
            idx_no_diversity=np.argmin(proto_dist_j, axis=None)
        

        if condition: #batch_min_proto_dist_j < global_min_proto_dist[j]
            if(diversity==False):
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(idx_no_diversity,
                                        proto_dist_j.shape))
            
            if(diversity==True):
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(idx_yes_diversity,
                                        proto_dist_j.shape))

            # if class_specific:
            #     '''
            #     change the argmin index from the index among
            #     images of the target class to the index in the entire search
            #     batch
            #     '''
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
           
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
            
            coordenadas_do_patch_latent_space=(fmap_height_start_index,fmap_width_start_index)
            idx_patch_no_mapa_latente=coordenadas_do_patch_latent_space[0]*protoL_input_.shape[2]+coordenadas_do_patch_latent_space[1]

            


            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            #More things added            
            from settings import img_size
            ni, L, H, W = protoL_input_.shape
            h_range = range(fmap_height_start_index, fmap_height_end_index)
            w_range = range(fmap_width_start_index, fmap_width_end_index)
            hh = h_range[0] * int(img_size / H)
            ww = w_range[0] * int(img_size / W)
            patch_og = original_img_j[hh:hh+int(img_size / H), ww:ww+int(img_size / W), :]
            # create a copy of the original image
            img_copy = original_img_j.copy()
            # get the coordinates of the bounding box
            x1 = ww
            y1 = hh
            x2 = x1 + int(img_size / W)
            y2 = y1 + int(img_size / H)
            # draw the bounding box on the image
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 1, 0), 2)
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]
            #More stuff i added
            proto_img_j_copy=original_img_j.copy()
            cv2.rectangle(proto_img_j_copy,(proto_bound_j[2], proto_bound_j[0]), 
                          (proto_bound_j[3], proto_bound_j[1]), (1, 1, 0), thickness=2)

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               cmap='gray')
                    
                    # write the ISIC ID info down
                    with open(os.path.join(dir_for_saving_prototypes,
                                           prototype_img_filename_prefix + '-lookup_ISIC_ID_' + str(j) + '.txt'), "w") as f:
                        f.write(str(ISIC_ID[img_index_in_batch])+'\nIDX_PATCH_IN_LATENT_MAP='+str(idx_patch_no_mapa_latente))
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            'prototype_in_img' + str(j) + '.png'),
                               proto_img_j_copy,
                               vmin=0.0,
                               vmax=1.0)
                    
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            'P_patch_' + str(j) + '.png'),
                               patch_og,
                               vmin=0.0,
                               vmax=1.0)
                    
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            'P_patch_in_ogI_' + str(j) + '.png'),
                               img_copy,
                               vmin=0.0,
                               vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict
