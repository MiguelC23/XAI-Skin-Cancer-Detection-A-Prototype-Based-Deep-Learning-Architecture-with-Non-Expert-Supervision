##### MODEL AND DATA LOADING
if __name__ == '__main__':
    import torch
    import torch.utils.data
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.autograd import Variable
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import re
    import os
    import copy
    from helpers import makedir, find_high_activation_crop
    from log import create_logger
    from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

    k=2 #topk classes
    most_activated_prototypes=9#chosse the same number as prototypes if you want.
    # specify the test image to be analyzed
    Test_Dataset_variable=1 # 0 ISIC 2019; 1 PH2; 2 DERM7PT
    MASKS_IN_TEST=False
    import random
    print("Choose 0 to analyze a random Melanoma image\nChoose 1 to analyze a random Nevus image")
    inp = ''
    valid_inputs = [0,1]
    output = {0: 'You decided to choose a MELANOMA image', 1:'You decided to choose an image of NEVUS'}
    while inp not in valid_inputs:
        inp = int(input("Enter 0 or 1: "))
        if inp not in valid_inputs:
            print("You must type 0 or 1")

    print(output[inp])

    if(inp==0):
        if(Test_Dataset_variable==0):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val\MEL"#args.test_img_dir[0]
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks\MEL"
        elif(Test_Dataset_variable==1):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_test\MEL"
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_TEST_FINE_MASKS\FINE_MASKS\MEL"
        elif(Test_Dataset_variable==2):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\derm7pt_like_ISIC2019\train_val_test_224\MEL"
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\DERM7PT_FINE_MASKS_224\MEL"

        test_image_name=random.choice(os.listdir(test_image_dir))
        #test_image_name='ISIC_0063496.jpg' 
        test_image_label = 0

    if(inp==1):
        if(Test_Dataset_variable==0):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val\NV"#args.test_img_dir[0]
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks\NV"
        elif(Test_Dataset_variable==1):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_test\NV"
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_TEST_FINE_MASKS\FINE_MASKS\NV"
        elif(Test_Dataset_variable==2):
            test_image_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\derm7pt_like_ISIC2019\train_val_test_224\NV"
            mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\DERM7PT_FINE_MASKS_224\NV"
        

        test_image_name=random.choice(os.listdir(test_image_dir)) 
        test_image_label = 1

    if(MASKS_IN_TEST==False):
        mask_dir=None
    else:
        filename = test_image_name
        new_filename = os.path.splitext(filename)[0]+".png"
        mask_path= os.path.join(mask_dir, new_filename)

    test_image_path = os.path.join(test_image_dir, test_image_name)
    print("test_image_path:",test_image_path)
    ##### MODEL AND DATA LOADING
    # load the model
    check_test_accu = False
                
    load_model_dir = r"C:\ACP_BM_Problem\NC2\resnet18\run1\\"#args.test_model_dir[0]  
    load_model_name = r"20_2push0.8295.pth"#args.test_model_name[0]

    save_analysis_path = os.path.join(load_model_dir, test_image_name)
    makedir(save_analysis_path)
    print('save_analysis_path',save_analysis_path)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number =int(epoch_number_str)

    log('load model from ' + load_model_path)
    log('test_image_dir: '+test_image_dir)
    log('test_image_name: '+test_image_name)
    log('test_image_path: '+test_image_path)
    

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(load_model_dir, 'img')

    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
    prototype_img_identity = prototype_info[:, -1]
    num_classes = len(set(prototype_img_identity))

    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    print(prototype_max_connection)
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        undo_preprocessed_img = undo_preprocess_input_function(img_copy)
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0]
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
        
        plt.imsave(fname, undo_preprocessed_img)
        return undo_preprocessed_img

    def save_prototype(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)
        
    def save_prototype_self_activation(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                        'prototype-img-original_with_self_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_original_img_with_bbox(fname, epoch, index,
                                            bbox_height_start, bbox_height_end,
                                            bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
        cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                    color, thickness=2)
        p_img_rgb = p_img_bgr[...,::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255
        #plt.imshow(p_img_rgb)
        #plt.axis('off')
        plt.imsave(fname, p_img_rgb)

    def save_P_patch_in_og_I_bboxgreen(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                        'P_patch_in_ogI_'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                        bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                    color, thickness=2)
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        #plt.imshow(img_rgb_float)
        #plt.axis('off')
        plt.imsave(fname, img_rgb_float)

    # load the test image and forward it through the network
    preprocess = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    normalize
    ])

    img_pil = Image.open(test_image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    mask_test=None
    if(MASKS_IN_TEST==True):
        mask= Image.open(mask_path).convert('L')

        def replace_values(x):
            return torch.where(x < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
        
        mask_transform =transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(replace_values)
            ])
        
        mask = mask_transform(mask)
        unique_values, counts = torch.unique(mask, return_counts=True)
        #IF MASKS HAS ONLY ONES POUT MASK WITH 0s to not penalize everything. So consider everything important.
        #Remember 0 is important. 1 is not important
        if(len(unique_values==1) and unique_values[0]==1 and counts[0]==img_size*img_size):
            mask=1-mask
        mask_test=Variable(mask.unsqueeze(0)).cuda()
        mask_test=mask_test.repeat(2, 1, 1, 1)

    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])

    logits, min_distances, upsampled_activation = ppnet_multi(images_test,mask_test)
    conv_output, distances = ppnet.push_forward(images_test,None)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    print(logits.shape)
    log('logits:'+str(logits))
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        log(str(i) + ' ' + str(tables[-1]))
    print(tables)
    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1] 
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))
    original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                        images_test, idx)

    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))
    max_act = 0
    log('Most activated %d most_activated_prototypes \n' % most_activated_prototypes) 
    log('prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
    for i in range(1,most_activated_prototypes+1):
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'top-%d_activated_prototype.png' % i),
                    start_epoch_number, sorted_indices_act[-i].item())
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                'top-%d_activated_prototype_in_original_pimg.png' % i),
                                            epoch=start_epoch_number,
                                            index=sorted_indices_act[-i].item(),
                                            bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                            bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                            bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                            bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                            color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                    'top-%d_activated_prototype_self_act.png' % i),
                                    start_epoch_number, sorted_indices_act[-i].item())
        
        save_P_patch_in_og_I_bboxgreen(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                    'top-%d_activated_Ppatch_in_ogI.png' % i),
                                    start_epoch_number, sorted_indices_act[-i].item())
        
        log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
        log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
        log('activation value (similarity score): {0}'.format(array_act[-i]))

        f = open(save_analysis_path + '/most_activated_prototypes/' + 'top-' + str(i) + '_activated_prototype.txt', "w")
        f.write('similarity: {0:.3f}\n'.format(array_act[-i].item()))
        f.write('last layer connection with predicted class: {0} \n'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        f.write('proto index:')
        f.write(str(sorted_indices_act[-i].item()) + '\n')
        for class_id_ in range(num_classes):
            f.write(f'proto connection to class {class_id_}:')
            f.write(str(ppnet.last_layer.weight[class_id_][sorted_indices_act[-i].item()]) + '\n')
        f.close()

        log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        
        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)


        # show the image overlayed with different normalized prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        
        # get the max activation of any proto on this image (works because we start with highest act, must be on rescale)
        if np.amax(rescaled_activation_pattern) > max_act:
            max_act = np.amax(rescaled_activation_pattern)

        rescaled_activation_pattern = rescaled_activation_pattern / max_act
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype_normed.png' % i),
                overlayed_img)
        log('--------------------------------------------------------------')

    log('***************************************************************')
    log('***************************************************************')
    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 2
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        proto_act_normed_maps = []
        makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

        log('top %d predicted class: %d' % (i+1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'top-%d_activated_prototype.png' % prototype_cnt),
                        start_epoch_number, prototype_index)
            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                epoch=start_epoch_number,
                                                index=prototype_index,
                                                bbox_height_start=prototype_info[prototype_index][1],
                                                bbox_height_end=prototype_info[prototype_index][2],
                                                bbox_width_start=prototype_info[prototype_index][3],
                                                bbox_width_end=prototype_info[prototype_index][4],
                                                color=(0, 255, 255))
            save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                        'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                        start_epoch_number, prototype_index)
            save_P_patch_in_og_I_bboxgreen(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'top-%d_activated_Ppatch_in_ogI.png' % prototype_cnt),
                        start_epoch_number, prototype_index)
            
            log('prototype index: {0}'.format(prototype_index))
            log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
            
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)
            
            # logging
            f = open(save_analysis_path + '/top-' + str(i+1) + '_class_prototypes/' + 'top-' + str(prototype_cnt) + '_activated_prototype.txt', "w")
            f.write('similarity: {0:.3f}\n'.format(prototype_activations[idx][prototype_index]))
            f.write('last layer connection: {0:.3f}\n'.format(ppnet.last_layer.weight[c][prototype_index]))
            f.write('proto index: ' + str(prototype_index) + '\n')
            for class_id_ in range(num_classes):
                f.write(f'proto connection to class {class_id_}:')
                f.write(str(ppnet.last_layer.weight[class_id_][prototype_index]) + '\n')
            f.close()

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                    high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                            img_rgb=original_img,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                    overlayed_img)

            # show the image overlayed with differently normed prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / max_act
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_top-%d_prototype_normed.png' % prototype_cnt),
                    overlayed_img)
            proto_act_normed_maps.append(rescaled_activation_pattern)
            log('--------------------------------------------------------------')
            prototype_cnt += 1

        class_heatmap = np.average(np.asarray(proto_act_normed_maps), axis=0)
        class_heatmap = class_heatmap - np.amin(class_heatmap)
        class_heatmap = class_heatmap / np.amax(class_heatmap)
        class_heatmap = cv2.applyColorMap(np.uint8(255*class_heatmap), cv2.COLORMAP_JET)
        class_heatmap = np.float32(class_heatmap) / 255
        class_heatmap = class_heatmap[...,::-1]
        # print("\n\n\n\nclass heatmap before avg", np.asarray(proto_act_normed_maps).shape)
        # print("\n\n\n\nclass heatmap after avg", np.average(np.asarray(proto_act_normed_maps), axis=0).shape)
        # print("\n\n\n\nclass heatmap min max", np.amin(class_heatmap), np.amax(class_heatmap))
        overlayed_img = 0.5 * original_img + 0.3 * class_heatmap
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_whole_%d_class.png' % c),
                    overlayed_img)
        log('***************************************************************')

    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()

