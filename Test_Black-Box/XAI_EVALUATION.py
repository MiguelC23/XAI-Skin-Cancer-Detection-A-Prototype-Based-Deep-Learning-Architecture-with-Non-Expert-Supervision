if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import torch
    import torch.utils.data
    import torch.nn as nn
    from log import create_logger
    import random
    from Dataset import PH2_Dataset_or_Derm7pt_XAI,ISIC2019_Dataset
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics._classification import confusion_matrix, precision_recall_fscore_support
    from tqdm import tqdm

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    import torchvision.transforms as transforms
    from preprocess import mean, std    
    import cv2
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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    random_seed_number=4
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    
    PH2=True
    test_masks_flag=True
    num_classes=2 # number f classes that the model has trained with
    load_model_path =r"E:\Coisas da Tese\Test_Black-Box\NC2\NC2_R18_OA_95_0.8372.pth"
    random_heatmaps_flag=False

    #This flags are important to show only results relate to the classes present in the Dataset we are testing
    #Because PH2 only has 2 classes and Derm7pt 6 classes
    #But if we trained the model with 8 classes need to pay attention to this flags
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False

    if(PH2==True):
        test_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_test"
        folder=r"E:\Coisas da Tese\Test_Black-Box\ResultsPH2"
    else:
        test_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\derm7pt_like_ISIC2019\train_val_test_224"
        folder=r"E:\Coisas da Tese\Test_Black-Box\Results_Derm7pt"

    # obter o diretório que contém o arquivo
    dir_path = os.path.dirname(load_model_path)

    # obter o nome do arquivo sem a extensão
    filename = os.path.splitext(os.path.basename(load_model_path))[0]

    # separar as partes do nome do arquivo que você precisa
    model_name = os.path.basename(os.path.dirname(load_model_path))

    # juntar o nome da pasta, nome do arquivo e extensão com um separador "______"
    if(PH2==True):
        testing_log_name = f"{model_name}_{filename}_XAI_PH2.txt"
    else:
        testing_log_name = f"{model_name}_{filename}_XAI_Derm7pt.txt"

    
    log, logclose = create_logger(log_filename=os.path.join(folder, testing_log_name))
    log("\t\t\tModel {}\n".format(testing_log_name))
    log(load_model_path)
    if(PH2==True):
        log("\t\t\tAnalysis of the PH2 dataset\n")
    else:
        log("\t\t\tAnalysis of the derm7pt dataset\n")

    model = torch.load(load_model_path)
    model = model.cuda()
    # Access the last convolutional layer
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module

    if last_conv_layer is None:
        raise ValueError("Could not find the last convolutional layer in the model.")
    target_layers = [last_conv_layer]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    model = torch.nn.DataParallel(model)

    
    test_batch_size = 10
    # test set
    if(test_masks_flag==True):
        if(PH2==True):
                test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_TEST_FINE_MASKS\FINE_MASKS"
        elif(PH2==False):
                test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\DERM7PT_FINE_MASKS_224"
    else:
        test_mask_dir=None
        
    test_dataset = PH2_Dataset_or_Derm7pt_XAI(test_dir, mask_dir=test_mask_dir, is_train=False,number_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    log('\t\t\ttest set size: {0}\n'.format(len(test_loader.dataset)))

    labels_test = [item[1] for item in test_dataset.ids]
    labels_test=np.array(labels_test)
    classes_test,counts= np.unique(labels_test,return_counts=True)
    log('\t\t\ttest labels: {0}\n'.format(classes_test))
    log('\t\t\ttest labels counts: {0}\n'.format(counts))

    # Validate
    log('\n\tTesting')
    all_predicted_val, all_target_val = [], []
    model.eval()
    n_batches_val=0
    #total_cross_entropy_val = 0
    device = torch.device("cuda")
    #with torch.no_grad():
    percentagens = np.arange(5, 105, 5).tolist()
    n_percentage=len(percentagens)
    all_predicted_p=[[] for _ in range(n_percentage)]
    all_target_p=[[] for _ in range(n_percentage)]
    contador=0
    fp_metric_cost=0
    for images, labels, images_ids, mask in test_loader:
        images, labels = images.to(device), labels.to(device)

        mascaras=mask.cuda()

        NORMALIZATION= transforms.Compose([transforms.Normalize(mean=mean, std=std)])
       
        input_no_normalized=images
        images = NORMALIZATION(images)
    
        outputs = model(images)


        n_batches_val += 1
        _, predicted = torch.max(outputs.data, 1)
        all_predicted_val.extend(predicted.detach().cpu().tolist())
        all_target_val.extend(labels.detach().cpu().tolist())
        
        numero_imagens,channels,H,W=images.shape
        for per in percentagens:
            all_target_p[percentagens.index(per)].extend(labels.detach().cpu().tolist())
        
        fp_metric_batch=0
        for i in tqdm(range(0,numero_imagens)):                 
            input_no_normalized_copia=input_no_normalized.clone()
            targets_cam = [ClassifierOutputTarget(predicted[i])]
            #print(images[i].shape)
            imagem_atual = images[i:i+1].clone().detach().requires_grad_()
            #print(imagem_atual.shape)
            if(random_heatmaps_flag==False):
                grayscale_cam = cam(input_tensor=imagem_atual, targets=targets_cam)
                grayscale_cam = grayscale_cam[0, :]
                #print(grayscale_cam.shape)
                ativacao_imagem_atual=torch.tensor(grayscale_cam) 
            else:                
                ativacao_imagem_atual=create_random_heatmap()
                grayscale_cam=np.array(ativacao_imagem_atual)
            contador+=1
            mascara_atual=torch.squeeze(mascaras[i,:,:,:]).to("cpu")
            fp_metric_batch+=torch.sum(mascara_atual*ativacao_imagem_atual)/torch.sum(ativacao_imagem_atual)
            r"""if(True):
                image_tensor=input_no_normalized[i,:,:,:]
                image_np = (image_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                # Transpose the dimensions from (3, 224, 224) to (224, 224, 3)
                rgb_img_np = image_np.transpose(1, 2, 0)
                rgb_img_np =np.float32(rgb_img_np ) / 255
                cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(r"E:\Coisas da Tese\Test_Black-Box\heatmaps\\"+"heatmap_"+images_ids[i]+"_"+str(contador)+".png", cam_image)"""
            for per in percentagens:
                num_pixels_remover = int((per / 100.0) * (images.shape[2] * images.shape[3]))
                indices_pixels_remover = torch.argsort(ativacao_imagem_atual.view(-1),descending=True)[:num_pixels_remover]
                coords_pixels_remover = torch.stack([indices_pixels_remover // ativacao_imagem_atual.size(1),
                                             indices_pixels_remover % ativacao_imagem_atual.size(1)], dim=1)
                
                input_no_normalized_copia[i,:, coords_pixels_remover[:, 0], coords_pixels_remover[:, 1]] = 0
                input_alterado_normalziado= NORMALIZATION(input_no_normalized_copia)
                r"""if(per==5):
                    image_tensor_removed_pixels = input_no_normalized_copia[i]
                    # Convert the torch tensor back to a NumPy array and scale it back to the range [0, 255]
                    # (assuming it was originally in the range [0.0, 1.0])
                    image_np_r = (image_tensor_removed_pixels * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    # Transpose the dimensions from (3, 224, 224) to (224, 224, 3)
                    image_np_r = image_np_r.transpose(1, 2, 0)
                    import matplotlib.pyplot as plt
                    # Display the image using matplotlib
                    plt.imshow(image_np_r)
                    plt.axis('off')  # Turn off axis ticks and labels
                    plt.savefig("output_image_"+str(per)+".png")"""
                    
            
                output_p  = model(input_alterado_normalziado[i:i+1,:,:,:])
                _, predicted_p = torch.max(output_p.data, 1)
                all_predicted_p[percentagens.index(per)].extend(predicted_p.detach().cpu().tolist())
                
        fp_metric_batch=fp_metric_batch/numero_imagens
        fp_metric_cost+=fp_metric_batch
    
    print(n_batches_val)            
    log('\t\t\t fp metric or MG4: {0}\n'.format(fp_metric_cost/n_batches_val))

    if(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        log('{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val,labels=[1, 2, 3,4,5,7]) ) ) )
        log('{0}'.format( classification_report(all_target_val, all_predicted_val,labels=[1, 2, 3,4,5,7]) ))
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        log('{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val))))
        log('{0}'.format( classification_report(all_target_val, all_predicted_val)))
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==True):
        log('{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val,labels=[4,5]) ) ) )
        log('{0}'.format( classification_report(all_target_val, all_predicted_val,labels=[4,5])))

    
    if(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target_val, all_predicted_val,
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target_val, all_predicted_val,labels=[1, 2, 3,4,5,7],
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==True):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target_val, all_predicted_val,labels=[4,5],
                                                        average='macro')
        
    recalls=np.zeros(n_percentage)
        
    print(np.array(all_predicted_p).shape)
    print(np.array(all_target_p).shape)
    if(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        for per in percentagens:
            _, recalls[percentagens.index(per)], _, _ = precision_recall_fscore_support(all_target_p[percentagens.index(per)], all_predicted_p[percentagens.index(per)],
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        for per in percentagens:
            _, recalls[percentagens.index(per)], _, _= precision_recall_fscore_support(all_target_p[percentagens.index(per)], all_predicted_p[percentagens.index(per)],labels=[1, 2, 3,4,5,7],
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==True):
        for per in percentagens:
            _, recalls[percentagens.index(per)], _, _ = precision_recall_fscore_support(all_target_p[percentagens.index(per)], all_predicted_p[percentagens.index(per)],labels=[4,5],
                                                        average='macro')
            
    bas_p_test=recalls
    if isinstance(bas_p_test, np.ndarray):
        bas_p_test = bas_p_test.tolist()
        bas_p_test.insert(0,rc)
        bas_p_test=[round(num, 3) for num in bas_p_test]
    log('\t\t\ttest BA LIST: {0}\n'.format(bas_p_test))
    percentagens.insert(0,0)
    log('\t\t\ttest PERCENTAGES LIST: {0}\n'.format(percentagens))



    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))




    if(num_classes==8):
        log('\n\tMalignant (1) vs Benign (0) information')
        BM_all_target = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
        BM_all_predicted = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
        for a in range(2):    
            log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target, BM_all_predicted)[a])))
        log('{0}'.format( classification_report(BM_all_target, BM_all_predicted) ))



