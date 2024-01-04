import heapq
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from helpers import makedir, find_high_activation_crop, dump, imsave_with_bbox
from receptive_field import compute_rf_prototype
from settings import img_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_heatmap(upsampled_act):
    """Overlay (upsampled) activation on original image and save the result"""

    rescaled_act_pattern = upsampled_act - np.amin(
        upsampled_act)
    rescaled_act_pattern = rescaled_act_pattern / np.amax(
        rescaled_act_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern),
                                cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    return heatmap


class ImagePatch:

    def __init__(self, patch, label, distance,
                 original_img=None, act_pattern=None, patch_indices=None, act_pr=None,patch_og=None,patch_og_in_img=None, patch_latent=None, patch_ID=None, patch_IDX=None):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices
        self.act_pr = act_pr
        self.patch_og=patch_og
        self.patch_og_in_img=patch_og_in_img
        self.patch_latent=patch_latent
        
        self.patch_ID=patch_ID
        self.patch_IDX=patch_IDX

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchInfo:

    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


def act_pr(activation_map_, fine_annotation, percentile: int):
    activation_map_ = cv2.resize(activation_map_,
                                 dsize=fine_annotation.shape,
                                 interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(activation_map_, percentile)
    mask = np.ones(activation_map_.shape)
    mask[activation_map_ < threshold] = 0
    assert fine_annotation.shape == mask.shape
    denom = np.sum(mask)
    num = np.sum(mask * fine_annotation)
    pr = num / denom
    return pr


def find_k_nearest_patches_to_prototypes(dataloader: DataLoader,
                                         # (must be unnormalized in [0,1])
                                         prototype_network_parallel,
                                         # pytorch network with prototype_vectors
                                         k: int = 5,
                                         preprocess_input_function=None,
                                         # normalize if needed
                                         full_save: bool = False,  # save all the images
                                         root_dir_for_saving_images: str = './nearest',
                                         log=print,
                                         prototype_activation_function_in_numpy=None):
    """
    Find the nearest patches in the dataset to each prototype

    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    """
    #numeros=0

    prototype_network_parallel.eval()
    log('find nearest patches')
    start = time.time()

    n_prototypes = prototype_network_parallel.module.num_prototypes
    prototype_shape = prototype_network_parallel.module.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])
 
    for idx, data in enumerate(dataloader):
        # search_batch_input [n_images, channel, img_size, img_size]
        search_batch_input = data[0]
        search_y = data[1]
        search_IDS=data[2]

        fine_annotations = None
        if len(data) > 3 and False:
            if torch.all(data[4]):
                fine_annotations = data[3]

        print(f'batch {idx}')

        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.to(device)
            protoL_input_torch, proto_dist_torch = \
                prototype_network_parallel.module.push_forward(search_batch,None)
        
        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        ni, L, H, W=protoL_input_.shape
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]

        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
        
        #if(numeros==0):
        #    np.save('distances'+str(numeros) +'.npy', proto_dist_ )
        #    np.save('features'+str(numeros) +'.npy', protoL_input_ )
        #    numeros=numeros+1

        # distance_map [n_protos, h, w]
        for img_idx, distance_map in enumerate(proto_dist_):
            for j in range(n_prototypes):
                # consider only images of the same class of j
                if not prototype_network_parallel.module.prototype_class_identity[
                    j, search_y[img_idx]]:
                    continue

                # find the closest patches in this batch to prototype j
                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])

                if full_save:
                    closest_patch_indices_in_distance_map_j = \
                        list(np.unravel_index(np.argmin(distance_map[j], axis=None),
                                              distance_map[j].shape))


                    fmap_height_start_index = closest_patch_indices_in_distance_map_j[0]*1
                    fmap_height_end_index = fmap_height_start_index + proto_h
                    fmap_width_start_index = closest_patch_indices_in_distance_map_j[1]*1
                    fmap_width_end_index = fmap_width_start_index + proto_w
                    h_range = range(fmap_height_start_index, fmap_height_end_index)
                    w_range = range(fmap_width_start_index, fmap_width_end_index)
                    hh = h_range[0] * int(img_size / H)
                    ww = w_range[0] * int(img_size / W)
                    
                    closest_patch_indices_in_distance_map_j = [
                                                                  0] + closest_patch_indices_in_distance_map_j
                    closest_patch_indices_in_img = \
                        compute_rf_prototype(search_batch.size(2),
                                             closest_patch_indices_in_distance_map_j,
                                             protoL_rf_info)
                    closest_patch = \
                        search_batch_input[img_idx, :,
                        closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
                        closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
                    #print(search_batch_input.shape)#RGB image [75,3,224,224] 75 is batch size, 224 is img size, 3 is RGB channels
                    latent_closest_patch=protoL_input_[img_idx, :, fmap_height_start_index:fmap_height_end_index, fmap_width_start_index:fmap_width_end_index]
                    #print( latent_closest_patch.shape)#[128,1,1] 128=prototype_shape[1] 
                    closest_patch = closest_patch.numpy()
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))
                    patch_ID= search_IDS[img_idx]
                    patch_IDX = fmap_height_start_index * H + fmap_width_start_index

                    original_img = search_batch_input[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))

                    patch_og = original_img[hh:hh+int(img_size / H), ww:ww+int(img_size / W), :]
                    # create a copy of the original image
                    img_copy = original_img.copy()
                    # get the coordinates of the bounding box
                    x1 = ww
                    y1 = hh
                    x2 = x1 + int(img_size / W)
                    y2 = y1 + int(img_size / H)
                    # draw the bounding box on the image
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 1, 0), 2)

                    if prototype_network_parallel.module.prototype_activation_function == 'log':
                        act_pattern = np.log((distance_map[j] + 1) / (distance_map[
                                                                          j] + prototype_network_parallel.module.epsilon))
                    elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                        act_pattern = max_dist - distance_map[j]
                    else:
                        act_pattern = prototype_activation_function_in_numpy(
                            distance_map[j])

                    pr = None
                    if fine_annotations is not None:
                        pr = act_pr(act_pattern, fine_annotations[img_idx].numpy(), 95)

                    # 4 numbers: height_start, height_end, width_start, width_end
                    patch_indices = closest_patch_indices_in_img[1:5]

                    # construct the closest patch object
                    closest_patch = ImagePatch(patch=closest_patch,
                                               label=search_y[img_idx],
                                               distance=closest_patch_distance_to_prototype_j,
                                               original_img=original_img,
                                               act_pattern=act_pattern,
                                               patch_indices=patch_indices,
                                               act_pr=pr,
                                               patch_og=patch_og,
                                               patch_og_in_img=img_copy,
                                               patch_latent=latent_closest_patch,
                                               patch_ID=patch_ID,
                                               patch_IDX=patch_IDX)
                else:
                    closest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   distance=closest_patch_distance_to_prototype_j)

                # add to the j-th heap
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], closest_patch)

    # after looping through the dataset every heap will
    # have the k closest prototypes
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if full_save:
            _save_images_in_heap(heaps, j, root_dir_for_saving_images)

    # fill remaining slots with -1 in order to have the same length for every prototype
    labels_all_prototype = []
    for j in range(n_prototypes):
        tmp = []
        for patch in heaps[j]:
            tmp.append(patch.label)
        for _ in range(k - len(tmp)):
            tmp.append(-1)
        labels_all_prototype.append(tmp)

    labels_all_prototype = np.array(labels_all_prototype)
    # labels_all_prototype = np.array(
    #     [[patch.label for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)

    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start))

    return labels_all_prototype


def _save_images_in_heap(heaps: list, j: int, root_dir_for_saving_images: str) -> None:
    dir_for_saving_images = os.path.join(root_dir_for_saving_images, str(j))
    makedir(dir_for_saving_images)
    for i, patch in enumerate(heaps[j]):
        # save the activation pattern of the original image where the patch comes from
        dump(os.path.join(dir_for_saving_images, f'nearest-{i + 1}_act.pickle'),
             (patch.act_pattern, patch.act_pr))
        
        #save patch_og that has always a fixed size not defined with the activation
        plt.imsave(fname=os.path.join(dir_for_saving_images,
                                      f'nearest-{i + 1}_ogPatch.png'),
                   arr=patch.patch_og, vmin=0.0, vmax=1.0)
        
        #save patch_og has bounding box in img that has always a fixed size not defined with the activation
        plt.imsave(fname=os.path.join(dir_for_saving_images,
                                      f'nearest-{i + 1}_ogPatch_inI.png'),
                   arr=patch.patch_og_in_img, vmin=0.0, vmax=1.0)
        
        # save the original image where the patch comes from
        plt.imsave(fname=os.path.join(dir_for_saving_images,
                                      f'nearest-{i + 1}_original.png'),
                   arr=patch.original_img, vmin=0.0, vmax=1.0)

        # overlay (upsampled) activation on original image and save the result
        img_size: int = patch.original_img.shape[0]
        upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                           dsize=(img_size, img_size),
                                           interpolation=cv2.INTER_CUBIC)

        heatmap = compute_heatmap(upsampled_act_pattern)

        overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
        plt.imsave(fname=os.path.join(dir_for_saving_images,
                                      f'nearest-{i + 1}_original_with_heatmap.png'),
                   arr=overlayed_original_img, vmin=0.0, vmax=1.0)
        
        np.save(os.path.join(dir_for_saving_images,
                                 f'nearest-{i + 1}_latent_patch_{patch.patch_ID}_{patch.patch_IDX}.npy'),
                    patch.patch_latent)

        # if different from original image, save the patch (i.e. receptive field)
        if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
            np.save(os.path.join(dir_for_saving_images,
                                 f'nearest-{i + 1}_receptive_field_indices.npy'),
                    patch.patch_indices)
            # plt.imsave(fname=os.path.join(dir_for_saving_images,
            #                               f'nearest-{i + 1}_receptive_field.png'),
            #            arr=patch.patch, vmin=0.0, vmax=1.0)

            # save the receptive field patch with heatmap
            overlayed_patch = overlayed_original_img[
                              patch.patch_indices[0]:patch.patch_indices[1],
                              patch.patch_indices[2]:patch.patch_indices[3], :]
            # plt.imsave(fname=os.path.join(dir_for_saving_images,
            #                               f'nearest-{i + 1}_receptive_field_with_heatmap.png'),
            #            arr=overlayed_patch, vmin=0.0, vmax=1.0)

        # save the highly activated patch
        high_act_patch_indices = find_high_activation_crop(upsampled_act_pattern)
        high_act_patch = patch.original_img[
                         high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]
        np.save(os.path.join(dir_for_saving_images,
                             f'nearest-{i + 1}_high_act_patch_indices.npy'),
                high_act_patch_indices)
        plt.imsave(fname=os.path.join(dir_for_saving_images,
                                      f'nearest-{i + 1}_high_act_patch.png'),
                   arr=high_act_patch,
                   vmin=0.0,
                   vmax=1.0)
        # save the original image with bounding box showing high activation patch
        imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                            f'nearest-{i + 1}_high_act_patch_in_og_img.png'),
                         img_rgb=patch.original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3],
                         color=(0, 255, 255))
    labels = np.array([patch.label for patch in heaps[j]])
    np.save(os.path.join(dir_for_saving_images, 'class_id.npy'), labels)
