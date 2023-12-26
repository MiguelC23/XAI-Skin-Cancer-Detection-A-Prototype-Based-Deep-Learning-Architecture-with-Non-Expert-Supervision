import os
import re
from collections import defaultdict
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from torch.utils.data.dataloader import DataLoader
import train_and_test as tnt

import find_nearest

from helpers import makedir
from model import PPNet
from preprocess import preprocess_input_function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          color=(0, 255, 255)):
    
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                        'prototype-img-original' + str(index) + '.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start),
                  (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    # plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def activation_precision(load_model_dir: str,
                         model: PPNet,
                         data_set: DataLoader,
                         epoch_number_str: int,
                         preprocess_input_function=None,
                         percentile: int = 95,
                         per_proto: bool = False):
    """Interpretability metric (IAIA-BL paper)"""

    print('Compute activation precision')
    n_prototypes = model.module.num_prototypes

    precisions = []
    per_proto_hp = defaultdict(list)

    for idx, data in enumerate(data_set):
        print('\tbatch {}'.format(idx))
        if True:
            with_fine_annotation = data[4]
            search_batch_input = data[0][with_fine_annotation]
            search_y = data[1][with_fine_annotation]
            fine_anno = data[3][with_fine_annotation]
            if len(search_y) == 0:
                print(f'Skip {idx}')
                continue
        else:
            search_batch_input = data[0]
            search_y = data[1]
            fine_anno = data[3]

        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.to(device)
            fine_anno = fine_anno.to(device)
            protoL_input_torch, proto_dist_torch = model.module.push_forward(
                search_batch)

        proto_acts = model.module.distance_2_similarity(proto_dist_torch)

        proto_acts = torch.nn.Upsample(
            size=(search_batch.shape[2], search_batch.shape[3]), mode='bilinear',
            align_corners=False)(proto_acts)

        # confirm prototype class identity
        load_img_dir = os.path.join(load_model_dir, 'img')

        prototype_info = np.load(os.path.join(load_img_dir,
                                              f'epoch-{epoch_number_str}',
                                              f'bb{epoch_number_str}.npy'))
        prototype_img_identity = prototype_info[:, -1]
        print('Prototypes are chosen from ' + str(
            len(set(prototype_img_identity))) + ' number of classes.')
        print('Their class identities are: ' + str(prototype_img_identity))

        proto_acts_ = np.copy(proto_acts.detach().cpu().numpy())
        fine_anno_ = np.copy(fine_anno.detach().cpu().numpy())
        assert proto_acts_.shape[0] == fine_anno_.shape[0]

        for img_idx, (activation_maps_per_proto, fine_annotation) in enumerate(
                zip(proto_acts_, fine_anno_)):
            # for every test img
            for j in range(n_prototypes):
                # for each proto
                if prototype_img_identity[j] == search_y[img_idx]:
                    # if proto class matches img class

                    activation_map = activation_maps_per_proto[j]
                    threshold = np.percentile(activation_map, percentile)
                    mask = np.ones(activation_map.shape)
                    mask[activation_map < threshold] = 0
                    mask = mask * activation_map
                    assert fine_annotation.shape == mask.shape
                    denom = np.sum(mask)
                    num = np.sum(mask * fine_annotation)
                    pr = num / denom

                    precisions.append(pr)
                    per_proto_hp[j].append(pr)

    if per_proto:
        per_proto_hp_list = []
        for k, v in per_proto_hp.items():
            per_proto_hp_list.append((k, v))
        per_proto_hp_list.sort(key=lambda x: x[0])
        return per_proto_hp_list
    else:
        return np.average(np.asarray(precisions))


def main(load_model_path: str, num_classes: int, k: int):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import random
    from settings import random_seed_number
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    load_model_dir = os.path.dirname(load_model_path)
    model_name = os.path.basename(load_model_path)
    epoch_iter = re.search(r'\d+(_\d+)?', model_name).group(0)
    start_epoch_number = re.search(r'\d+', epoch_iter).group(0)

    print('load model from ' + load_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.to(device)
    ppnet_multi = torch.nn.DataParallel(ppnet)
    ppnet_multi.eval()

    load_img_dir = os.path.join(load_model_dir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}',
                                          f'bb{start_epoch_number}.npy'))


    from Dataset import ISIC2019_Dataset
    train_push_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224_sum_all_concept_agree3\images_for_this_masks\MEL_NV"
    train_mask_dir=None
    train_push_batch_size=75
    train_push_dataset = ISIC2019_Dataset(train_push_dir, train_mask_dir, is_train=True,is_push=True,number_classes=num_classes)
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    
    data=train_push_loader
    suffix='_nearest_from_EDEASD'


    root_dir_for_saving_images = load_model_path.split('.pth')[0] + suffix
    makedir(root_dir_for_saving_images)

    find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=data,  # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi,
        # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function,  # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_images,
        log=print)
    

    # save prototypes in original images

    for j in range(ppnet.num_malignant_prototypes):
        makedir(os.path.join(root_dir_for_saving_images, str(j)))
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_images, str(j),
                                'prototype_in_original_pimg.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255))
    



if __name__ == '__main__':
    load_model_path=r"C:\1CP_BinaryProblem\NC2\resnet18\run1\20_0push0.7869.pth"
    k=10 # Number of images/patches closer to each prototype
    num_classes=2
    main(load_model_path,num_classes=num_classes, k=k)
