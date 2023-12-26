
def update_and_add_tsne_plot(x, tsne, ax, label, prototypes_malignant, num_malignant_prototypes):
    # Reshape the input tensor and convert to numpy
    N, C, H, W = x.shape
    x_reshaped = x.view(N, -1).cpu().numpy()

    # Combine data points and prototypes into a single array
    combined_data = np.vstack((x_reshaped, prototypes_malignant.view(num_malignant_prototypes, -1).cpu().numpy()))

    # Apply t-SNE transformation to the combined data
    #tsne = TSNE(n_components=2)
    combined_tsne = tsne.fit_transform(combined_data)

    # Create a color array based on the labels
    colors = ['blue' if l == 1 else 'red' for l in label]

    # Update the scatter plot with the new t-SNE points and colors for data points
    ax.scatter(combined_tsne[:N, 0], combined_tsne[:N, 1], c=colors, marker='o', alpha=0.05) #alpha=0.05, 0,1, 0.5

    # Create colors for prototypes
    number_colors_repeat = num_malignant_prototypes
    prototype_colors = ["red"] * number_colors_repeat #only malignant prototypes

    # Update the scatter plot with the new t-SNE points and colors for prototypes
    ax.scatter(combined_tsne[N:N+number_colors_repeat, 0], combined_tsne[N:N+number_colors_repeat, 1], c=prototype_colors, marker='x')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    # Set labels for the legend
    mel_label = 'MEL'
    nv_label = 'NV'

    # Create a legend with custom labels and colors
    legend_labels = {'MEL': 'red', 'NV': 'blue'}
    legend_elements = [plt.Line2D([0], [0], color=legend_labels['MEL'], lw=2, label=mel_label),
                    plt.Line2D([0], [0], color=legend_labels['NV'], lw=2, label=nv_label)]

    # Add legend with custom handles
    ax.legend(handles=legend_elements, loc='upper right')


def see_cluster(model, dataloader,tsne=None, ax=None, images=True,filter_patches=True,only_max_pool_patches=True, size=(7,7)):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    print('\n\ttest')    
    model.eval()
    x_big_list = []
    label_big_all_list=[]
   
    x_list=[]
    label_list=[]


    for i, (image, label,ID,mask) in enumerate(dataloader):                
        input = image.cuda()
        target = label.cuda()
        mascaras=mask.cuda()

        # Resize the mask to torch.Size([batchsize, 1, HZ, WZ])
        resized_mask = F.interpolate(mascaras, size=size, mode='nearest')

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req =torch.no_grad()
        with grad_req:
            x = model.module.conv_features(input)
            batch_size, D, P, P_same=x.shape
            distances_mel=model.module.prototype_distances(input)[:,0:model.module.num_malignant_prototypes,:,:]
            # Reshape the tensor to make it compatible with torch.argmin
            reshaped_distances = distances_mel.view(batch_size, model.module.num_malignant_prototypes, -1)  # Reshape to (BATCH SIZE,NUMBER OF MALIGNANT PROTOTYPES, HZ*WZ) 
            # Perform min pool operation along the last dimension (HZ*WZ elements in each [HZ, WZ] map)
            min_indices = torch.argmin(reshaped_distances, dim=2)

            # Now min_indices has the indices of the minimum values along the [HZ, WZ] dimensions
            # min_indices has shape torch.Size([batch_size, NUMBER OF MALIGNANT PROTOTYPES])
            # Create a tensor with row indices
            row_indices = torch.arange(min_indices.size(0)).unsqueeze(1).cuda()
            # Add row indices to each element in the original tensor
            result_tensor = min_indices + row_indices*(P*P)
            # Now, result_tensor contains the sum of row indices for each element
            indices_flatten=result_tensor.reshape(-1)
            indices_flatten_not_repeated_patches= torch.unique(indices_flatten)

            x_list.append(x.cpu())
            label_list.append(label.cpu())
            mask_zero = (resized_mask == 0).reshape(batch_size * P * P)
            # Create a tensor with zeros of the same length as mask_zero
            only_masked_and_maxpool_patches = torch.zeros_like(mask_zero, dtype=torch.bool)
            # Find the indices that satisfy the conditions
            valid_indices = indices_flatten_not_repeated_patches[mask_zero[indices_flatten_not_repeated_patches]]
            # Set the corresponding indices in only_masked_and_maxpool_patches to True
            only_masked_and_maxpool_patches[valid_indices] = True
            x_big = x.permute(0, 2, 3, 1).reshape(batch_size * P * P, D, 1, 1)
            # Create label_big by repeating the labels
            label_big = np.repeat(label, P * P)
            if(filter_patches==True and images==False):
                if(only_max_pool_patches==True):
                    x_big=x_big[only_masked_and_maxpool_patches]
                    label_big=label_big[only_masked_and_maxpool_patches]
                else:
                    x_big=x_big[mask_zero]
                    label_big=label_big[mask_zero]
            x_big_list.append(x_big.cpu())
            label_big_all_list.append(label_big.cpu())

            


    # Concatenate all the x_big tensors on the CPU
    if(images==False): #each point is an image patch 
        x_big_all = torch.cat(x_big_list, dim=0)
        label_big_all=torch.cat(label_big_all_list, dim=0)
        num_prototypes_malginant=model.module.num_malignant_prototypes
        update_and_add_tsne_plot(x_big_all, tsne, ax,label_big_all,model.module.prototype_vectors[0:num_prototypes_malginant,:,:,:],num_malignant_prototypes=num_prototypes_malginant)
    elif(images==True): #each point is an image
        x_all=torch.cat(x_list, dim=0)
        label_all=torch.cat(label_list, dim=0)
        prototypes_mal=model.module.prototype_vectors[0:model.module.num_malignant_prototypes,:,:,:].cpu()

        # Initialize a list to store the indices of the most similar images for each prototype
        most_similar_image_indices = []

        # Set of used image indices to keep track of which images have already been assigned to prototypes
        used_image_indices = set()

        # Iterate through prototypes to find the most similar image (from different images) for each prototype
        proto_labels = 0 #malignant prototypes have label 0 MEL is 0 and NV is 1
        contador=-1
        for prototype in prototypes_mal:
            contador+=1
            min_similarity = float('inf')  # Initialize with positive infinity
            most_similar_index = -1  # Initialize with an invalid index

            for i, image in enumerate(x_all):
                if(label_all[i]==proto_labels):
                    # Check if this image index has already been used for another prototype
                    if i in used_image_indices:
                        continue

                    for x1 in range(P):
                        for x2 in range(P):
                            # Compute the mean squared error (MSE) as a similarity metric
                            mse = torch.mean((prototype[:D, 0, 0] - image[:D, x1, x2]) ** 2)

                            # Check if this image is more similar than the previous minimum
                            if mse < min_similarity:
                                min_similarity = mse
                                most_similar_index = i

            if most_similar_index >= 0:
                most_similar_image_indices.append(most_similar_index)

                # Mark the chosen image index as used
                used_image_indices.add(most_similar_index)
        

        images_of_p=[]
        for idx in most_similar_image_indices:
            image_now=x_all[idx]
            images_of_p.append(image_now)

        images_of_p_torch=torch.stack(images_of_p)
        update_and_add_tsne_plot(x_all, tsne, ax,label_all,images_of_p_torch,model.module.num_malignant_prototypes)
            
    





if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    import torch
    import torch.utils.data
    matplotlib.use('TkAgg')  
    import random
    from sklearn.manifold import TSNE
    from Dataset import ISIC2019_Dataset
    from settings import random_seed_number
    import torch.nn.functional as F
    from settings import train_dir,train_push_dir
    from settings import train_mask_dir,online_augmentation

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    # train set
    
    images=False # If true each point is an image else each point is a patch
    filter_patches=True #  If True We only want relevant patchs marked with 0s in inverted masks. Only put True when images=False
    only_max_pool_patches=True # Select the most activated patch between an image and each melanoma prototype when True, as utilizing all patches in t-SNE from the training dataset proves to be computationally expensive.
    load_model_path=r"C:\1CP_BinaryProblem\NC2\resnet18\run1\20_0push0.7869.pth"
    size=(7,7) # Dimension of the maps that are the input of the prototype layer, i.e the output of convolution layers. Example [BATCH_SIZE,D,P,P]. So you should put size=(P,P). Only VGG16 has (14,14) the others is (7,7)

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    num_classes=ppnet_multi.module.num_classes

    train_dataset = ISIC2019_Dataset(train_push_dir, train_mask_dir, is_train=True,number_classes=num_classes,augmentation=online_augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=250, shuffle=True,
        num_workers=4, pin_memory=False)

    #Create a t-SNE model
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,init="pca",learning_rate="auto") #30

    # Create an empty scatter plot
    fig, ax = plt.subplots()
    see_cluster(model=ppnet_multi, dataloader=train_loader,tsne=tsne,ax=ax,images=images,filter_patches=filter_patches,only_max_pool_patches=only_max_pool_patches,size=size)
    plt.show()






