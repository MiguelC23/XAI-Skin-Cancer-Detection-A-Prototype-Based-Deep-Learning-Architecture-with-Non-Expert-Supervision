if __name__ == '__main__':
    import os
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import torch
    import torch.utils.data
    # import torch.utils.data.distributed
    #from dataHelper import DatasetFolder
    from helpers import makedir
    import test_other_dataset as tnt
    from log import create_logger
    import random
    from Dataset import PH2_Dataset_or_Derm7pt,ISIC2019_Dataset
    from sklearn.utils.class_weight import compute_class_weight
    from settings import train_mask_dir,train_dir


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    random_seed_number=4
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

   
    caminho=r"C:\ACP_BM_Problem\NC8\resnet18\run1\5push0.4254.pth" 
    idx_P_a_remover=1
    stringP='removeP_'+str(idx_P_a_remover)
    VALMASKS=False
    MASKS_IN_TEST_SETS=False
    PH2=False 
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False

    VAL_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val"
    if(VALMASKS==True):
        VAL_masks_path=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks"
    else:
        VAL_masks_path=None

    load_model_path=caminho
    print(load_model_path)

    if(PH2==True):
        test_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_test"
        folder=r"C:\ACP_BM_PROBLEM\PH2_Results"
    else:
        test_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\derm7pt_like_ISIC2019\train_val_test_224"
        folder=r"C:\ACP_BM_PROBLEM\Derm7pt_Results"

    # obter o diretório que contém o arquivo
    dir_path = os.path.dirname(load_model_path)

    # obter o nome do arquivo sem a extensão
    filename = os.path.splitext(os.path.basename(load_model_path))[0]

    # separar as partes do nome do arquivo que você precisa
    model_name = os.path.basename(os.path.dirname(load_model_path))

    # juntar o nome da pasta, nome do arquivo e extensão com um separador "______"
    if(PH2==True):
        testing_log_name = f"{model_name}_{filename}_{stringP}_PH2.txt"
    else:
        testing_log_name = f"{model_name}_{filename}_{stringP}_Derm7pt.txt"

    log, logclose = create_logger(log_filename=os.path.join(folder, testing_log_name))
    log("\t\t\tModel {}\n".format(testing_log_name))
    log(load_model_path)
    if(PH2==True):
        log("\t\t\tAnalysis of the PH2 dataset\n")
    else:
        log("\t\t\tAnalysis of the derm7pt dataset\n")

    coefs = {
    'crs_ent': 1,
    }
    

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    with torch.no_grad():
            for class_i in range(ppnet.num_classes):
                ppnet.last_layer.weight[class_i][idx_P_a_remover]=0
    log('\nlast layer weights:\n {0}\n'.format(ppnet.last_layer.weight))        
    ppnet_multi = torch.nn.DataParallel(ppnet)
    num_classes=ppnet_multi.module.num_classes
    log("\nNUMBER OF CLASSES OF LOADED MODEL: {0}\n".format(num_classes))


    train_batch_size = 75
    train_dataset = ISIC2019_Dataset(train_dir, train_mask_dir, is_train=True,number_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    # Get the list of class labels
    labels_train = [item[1] for item in train_dataset.ids]
    labels_train=np.array(labels_train)
    # Compute the balanced weights
    weights = compute_class_weight(class_weight='balanced',classes= np.unique(labels_train), y=labels_train)
    # Print the balanced weights for each class
    print("Cross entropy balanced weights")
    for i, weight in enumerate(weights):
        print(f"Class {train_dataset.classes[i]}: {weight:.2f}")
    print("\n")
    

    log("\n========================Validation Set Performance========================\n")
    VAL_dataset = ISIC2019_Dataset(VAL_dir, mask_dir=VAL_masks_path, is_train=False,number_classes=num_classes)
    VAL_loader = torch.utils.data.DataLoader(
        VAL_dataset, batch_size=75, shuffle=False,
        num_workers=4, pin_memory=False)
    
    log('\t\t\tval set size: {0}\n'.format(len(VAL_loader.dataset)))
    labels_val = [item[1] for item in VAL_dataset.ids]
    labels_val=np.array(labels_val)
    classes_VAL,counts_VAL= np.unique(labels_val,return_counts=True)
    log('\t\t\tvalidayion labels: {0}\n'.format(classes_VAL))
    log('\t\t\tvalidation labels counts: {0}\n'.format(counts_VAL))

    accu_test_og,F1_score_test_og,ba_test_og = tnt.test(model=ppnet_multi, dataloader=VAL_loader, coefs=coefs,log=log, weights=weights)
    
    log("\n========================Test Set Performance========================\n")
    test_batch_size = 100
    # test set
    if(MASKS_IN_TEST_SETS==True):
        if(PH2==True):
            test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\PH2_TEST_FINE_MASKS\FINE_MASKS"
        elif(PH2==False):
            test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\DERM7PT_FINE_MASKS_224"
    else:
        test_mask_dir=None
        
    test_dataset = PH2_Dataset_or_Derm7pt(test_dir, mask_dir=test_mask_dir, is_train=False,number_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    log('\n\n\n\n\t\t\ttest set size: {0}\n'.format(len(test_loader.dataset)))

    labels_test = [item[1] for item in test_dataset.ids]
    labels_test=np.array(labels_test)
    classes_test,counts= np.unique(labels_test,return_counts=True)
    log('\t\t\ttest labels: {0}\n'.format(classes_test))
    log('\t\t\ttest labels counts: {0}\n'.format(counts))
    

    accu_test,F1_score_test,ba_test = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs, log=log, weights=weights,
                                               FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT,
                                               FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2)

    

