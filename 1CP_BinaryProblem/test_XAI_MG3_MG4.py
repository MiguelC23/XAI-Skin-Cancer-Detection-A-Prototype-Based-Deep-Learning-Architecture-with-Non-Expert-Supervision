import time
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support


import torchvision.transforms as transforms
from preprocess import mean, std



# Função para calcular a média ponderada dos mapas de ativação com base nas máscaras
def feature_portion_metric(upsampled_activation, mascaras, model,output):
    num_images = upsampled_activation.size(0)
    total=0
    for i in range(num_images):
        pesos_layer_for_label=model.module.last_layer.weight[:]*(-1)
        mapas_imagem_atual=upsampled_activation[i]
        mapas_imagem_atual=mapas_imagem_atual[0:model.module.num_malignant_prototypes,:,:]
        pesos_layer_for_label_expandidos = pesos_layer_for_label.view(-1, 1, 1)
        mapa_pesado= torch.sum(mapas_imagem_atual * pesos_layer_for_label_expandidos, dim=0)
        ativacao_imagem_atual = mapa_pesado.clone()
        valor_minimo = torch.min(ativacao_imagem_atual)
        valor_maximo = torch.max(ativacao_imagem_atual)
        ativacao_imagem_atual = (ativacao_imagem_atual - valor_minimo) / (valor_maximo - valor_minimo)
        
        mascara_atual=mascaras[i]
        mascara_atual=mascara_atual.squeeze(0)
        valor= torch.sum(mascara_atual*ativacao_imagem_atual)/torch.sum(ativacao_imagem_atual)
        total+=valor


    fpm_batch=total/num_images
    return fpm_batch


import cv2
import torch.nn as nn
def gaussian_kernel(size, sigma, type='Sum'):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    #print("kernel shape {}".format(kernel.shape))
    if type=='Sum':
      kernel = kernel / kernel.sum()
    else:
      kernel = kernel / kernel.max()
    return kernel.astype('double')

def create_overlayed_image(original_img, upsampled_activation_img):
    # Normalize the activation image to have values between 0 and 1
    rescaled_activation_img = upsampled_activation_img - np.amin(upsampled_activation_img)
    rescaled_activation_img = rescaled_activation_img / np.amax(rescaled_activation_img)
    rescaled_activation_img_og=rescaled_activation_img
    
    Kernel_size=32 
    sigma= 8 #Kernel_size/8.5
    hm = gaussian_kernel(Kernel_size,sigma)
    X  = cv2.filter2D(rescaled_activation_img,-1,hm)
    heatmap=X
    heatmap = (heatmap -heatmap.min()) / (heatmap.max() - heatmap.min())
    rescaled_activation_img=heatmap                                

    heatmap_3_channels = np.expand_dims(rescaled_activation_img, axis=-1)
    heatmap_3_channels = np.repeat(heatmap_3_channels, 3, axis=-1)

    # Apply the Jet color map to the 3-channel heatmap
   
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_3_channels), cv2.COLORMAP_JET)

    # Resize the heatmap to match the dimensions of the original image
    heatmap_colored = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))

    # Create the overlayed image
    overlayed_original_img = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.3, 0)

    #Original withouth gausssian filter to compare
    heatmap_3_channels_og= np.expand_dims(rescaled_activation_img_og, axis=-1)
    heatmap_3_channels_og= np.repeat(heatmap_3_channels_og, 3, axis=-1)
    # Apply the Jet color map to the 3-channel heatmap
    heatmap_colored_og= cv2.applyColorMap(np.uint8(255 * heatmap_3_channels_og), cv2.COLORMAP_JET)
    # Resize the heatmap to match the dimensions of the original image
    heatmap_colored_og= cv2.resize(heatmap_colored_og, (original_img.shape[1], original_img.shape[0]))
    # Create the overlayed image
    overlayed_original_img_og= cv2.addWeighted(original_img, 0.5, heatmap_colored_og, 0.3, 0)
    #cv2.imwrite("heatmap_og"+".png", overlayed_original_img_og)

    return overlayed_original_img


from scipy.ndimage import gaussian_filter
def create_smooth_heatmap(size, num_peaks=5, peak_std=30, normalize=True):
    heatmap = np.zeros(size, dtype=np.float32)
    peak_positions = np.random.randint(0, size[0], size=(num_peaks, 2))
    for position in peak_positions:
        heatmap[position[0], position[1]] = 1.0
    heatmap_smoothed = gaussian_filter(heatmap, sigma=peak_std)
    if normalize:
        heatmap_smoothed = (heatmap_smoothed - np.min(heatmap_smoothed)) / (np.max(heatmap_smoothed) - np.min(heatmap_smoothed))
    return heatmap_smoothed

def create_random_heatmap():
    # Tamanho do heatmap
    size = (224, 224)
    # Cria o heatmap suave com 5 picos e desvio padrão de 30
    heatmap_smooth = create_smooth_heatmap(size, num_peaks=5, peak_std=30)
    heatmap_smooth=torch.tensor(heatmap_smooth)
    return heatmap_smooth


def _train_or_test(model, dataloader, optimizer=None,coefs=None, log=print,weights=None,heatmap_random=False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None 
    start = time.time()
    n_batches = 0

    total_all_cost=0
    total_cross_entropy = 0
    fp_metric_cost=0
    all_predicted, all_target = [], []
    percentagens = np.arange(5, 105, 5).tolist()
    n_percentage=len(percentagens)
    all_predicted_p=[[] for _ in range(n_percentage)]
    all_target_p=[[] for _ in range(n_percentage)]
 
    for i, (image, label,ID,mask) in enumerate(dataloader):

        NORMALIZATION= transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        input = NORMALIZATION(image.cuda())
        input_no_normalized=image.cuda()
        target = label.cuda()
        mascaras=mask.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances, upsampled_activation, _ = model(input,mascaras)
            fp_metric=feature_portion_metric(upsampled_activation,mascaras,model,output)
            if(heatmap_random==True):
                heatmaps_aleatorios=[]
                fp_metric=0
                for aa in range(0,upsampled_activation.size(0)):
                    heatmaps_aleatorios.append(create_random_heatmap())
                    mascara_atual=torch.squeeze(mascaras[i,:,:,:]).to("cpu")
                    ativacao_atual=heatmaps_aleatorios[aa]
                    fp_metric+=torch.sum(mascara_atual*ativacao_atual)/torch.sum(ativacao_atual)
                fp_metric=fp_metric/upsampled_activation.size(0)

            num_images = upsampled_activation.size(0)
            for per in percentagens:
                num_pixels_remover = int((per / 100.0) * (image.shape[2] * image.shape[3]))                
                input_no_normalized_copia=input_no_normalized.clone()
                for i in range(num_images):
                    pesos_layer_for_label=model.module.last_layer.weight[:]*(-1)
                    mapas_imagem_atual=upsampled_activation[i]
                    mapas_imagem_atual=mapas_imagem_atual[0:model.module.num_malignant_prototypes,:,:]
                    pesos_layer_for_label_expandidos = pesos_layer_for_label.view(-1, 1, 1)
                    mapa_pesado= torch.sum(mapas_imagem_atual * pesos_layer_for_label_expandidos, dim=0)
                    ativacao_imagem_atual = mapa_pesado.clone()
                    valor_minimo = torch.min(ativacao_imagem_atual)
                    valor_maximo = torch.max(ativacao_imagem_atual)
                    ativacao_imagem_atual = (ativacao_imagem_atual - valor_minimo) / (valor_maximo - valor_minimo)
                    if(heatmap_random==True):
                        ativacao_imagem_atual=heatmaps_aleatorios[i]

                    indices_pixels_remover = torch.argsort(ativacao_imagem_atual.view(-1),descending=True)[:num_pixels_remover]
                    coords_pixels_remover = torch.stack([indices_pixels_remover // ativacao_imagem_atual.size(1),
                                             indices_pixels_remover % ativacao_imagem_atual.size(1)], dim=1)

                    input_no_normalized_copia[i,:, coords_pixels_remover[:, 0], coords_pixels_remover[:, 1]] = 0

                    #Uncoment to save heatmaps from the images that are used to calculate the metrics
                    """if(True):
                        image_tensor = input_no_normalized_copia[i]
                        # Convert the torch tensor back to a NumPy array and scale it back to the range [0, 255]
                        # (assuming it was originally in the range [0.0, 1.0])
                        image_np = (image_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        # Transpose the dimensions from (3, 224, 224) to (224, 224, 3)
                        image_np = image_np.transpose(1, 2, 0)
                        import matplotlib.pyplot as plt
                        # Display the image using matplotlib
                        plt.imshow(image_np)
                        plt.axis('off')  # Turn off axis ticks and labels
                        #plt.savefig("output_image_"+str(per)+".png")
                        if(per==5):
                            ativacao_imagem_atual_numpy= ativacao_imagem_atual.detach().cpu().numpy()
                            imagem_og=input_no_normalized[i]
                            imagem_og_np=(imagem_og * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                            imagem_og_np = imagem_og_np.transpose(1, 2, 0)
                            heatmap_imagem=create_overlayed_image(imagem_og_np,ativacao_imagem_atual_numpy)
                            numero=i+1+n_batches*num_images
                            cv2.imwrite(r"C:\1CP_BinaryProblem\heatmaps\\"+"heatmap_"+str(numero)+".png", heatmap_imagem)"""

                    input_alterado_normalziado= NORMALIZATION(input_no_normalized_copia)

                    
                output_p, min_distances_p, upsampled_activation_p ,_ = model(input_alterado_normalziado,mascaras)
                predicted_p = torch.round(torch.sigmoid(output_p)).long()
                all_predicted_p[percentagens.index(per)].extend(predicted_p.detach().cpu().tolist())
                all_target_p[percentagens.index(per)].extend(target.detach().cpu().tolist())


            # compute loss
            #ISIC 2019
            class_weights = torch.FloatTensor(weights).cuda()
            pos_weight = class_weights[1] / class_weights[0]  # Set pos_weight to the weight of the positive class
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            cross_entropy = criterion(output, target.float())

            # evaluation statistics
            # _, predicted = torch.max(output.data, 1)
            predicted = torch.round(torch.sigmoid(output)).long()
            all_predicted.extend(predicted.detach().cpu().tolist())
            all_target.extend(target.detach().cpu().tolist())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            fp_metric_cost += fp_metric
            
            total_all_cost+= (coefs['crs_ent'] * cross_entropy)

        # compute gradient and do SGD step
        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\ttotal cost:\t{0}'.format(total_all_cost / n_batches))
    log('\tfp metric and what we call MG4:\t{0}'.format(fp_metric_cost/n_batches))
      
    log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted))))
    log('{0}'.format( classification_report(all_target, all_predicted)))

    pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,average='macro')
    accu=accuracy_score(all_target, all_predicted)


    recalls=np.zeros(n_percentage)
    for per in percentagens:
        _, recalls[percentagens.index(per)], _, _ = precision_recall_fscore_support(all_target_p[percentagens.index(per)], all_predicted_p[percentagens.index(per)],average='macro')

    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))
            
    return accu,f1,rc,recalls,percentagens


def test(model, dataloader, coefs=None, log=print,weights=None,heatmap_random=False):
    log('\n\ttest')    
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,coefs=coefs, log=log, weights=weights,heatmap_random=heatmap_random)



    



    
