if __name__ == '__main__':
    import os
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import torch
    import torch.utils.data
    # import torch.utils.data.distributed
    import train_and_test as tnt
    from log import create_logger
    import random
    from Dataset import ISIC2019_Dataset
    from settings import random_seed_number
    from settings import train_dir, test_dir, \
                        train_batch_size, test_batch_size
    from settings import train_mask_dir
    from sklearn.utils.class_weight import compute_class_weight


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    load_model_path=r"C:\ACP_BM_Problem\NC2\resnet18\run1\20_2push0.8295.pth"
    folder_path_to_save_results=r"C:\ACP_BM_Problem\Weights_of_P"
    MASKS_IN_TEST_SETS=True
    if(MASKS_IN_TEST_SETS==True):
        test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks"#r"C:\PATH\TO\VALIDATION\MASKS"
    else:
        test_mask_dir=None

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    num_classes=ppnet_multi.module.num_classes
    num_prototypes=ppnet_multi.module.num_prototypes

    # train set is only gonna be used to obtain the class weights because it is given as input to the cross entropy loss during train and test
    train_dataset = ISIC2019_Dataset(train_dir, train_mask_dir, is_train=True,number_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    
    # test set
    test_dataset = ISIC2019_Dataset(test_dir, mask_dir=test_mask_dir, is_train=False,number_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
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

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,#0.8 
        'sep': -0.08,#-0.08
        'l1': 1e-4,
        'LM': 0,#0.001 or #0.0001
        'LF':0,#0.09
        'LR':0,#0.02
    }

    dir_path = os.path.dirname(load_model_path)
    # get the file name without the extension
    filename = os.path.splitext(os.path.basename(load_model_path))[0]
    # separate the parts of the file name that you need
    model_name = os.path.basename(os.path.dirname(load_model_path))
    # Combine the folder name, file name, and extension with a separator "______".
    testing_log_name = f"{model_name}_{filename}.txt"

    log, logclose = create_logger(log_filename=os.path.join(folder_path_to_save_results, testing_log_name))
    log('RESULTS ORIGINAL MODEL\n')
    accu_test_og,F1_score_test_og,ba_test_og = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs, log=log, wandb=print, weights=weights)
    bas_diff_list=[]
    bas_list=[]
    for i in range(num_prototypes):
        ppnet_og = torch.load(load_model_path)
        ppnet_og = ppnet_og.cuda()
        with torch.no_grad():
            for class_i in range(num_classes):
                ppnet_og.last_layer.weight[class_i][i]=0

        log("{}".format(ppnet_og.last_layer.weight))    
        ppnet_multi_alterar= torch.nn.DataParallel(ppnet_og)
        log("\n========================PROTOTYPE {} ABSENT ========================\n".format(i))
        accu_test,F1_score_test,ba_test = tnt.test(model=ppnet_multi_alterar, dataloader=test_loader, coefs=coefs,log=log,wandb=print, weights=weights)
        bas_list.append(ba_test)
        bas_diff_list.append(ba_test-ba_test_og)
        log("NEW BA {}\n".format(ba_test))
        log("Difference NEW_BA-BA_OG={}\n".format(ba_test-ba_test_og))
    
    lista_abs_diff = [abs(numero) for numero in bas_diff_list]
    indice_maximo = lista_abs_diff.index(max(lista_abs_diff))
    log("\nThe prototype that, when absent, had the greatest impact was {}, causing a performance difference of {}\n".format(indice_maximo,bas_diff_list[indice_maximo]))
    log("\n========================Resume========================\n")
    for i in range(num_prototypes):
        log("Prototype {} Absent".format(i))
        log("NEW_BA={}".format(bas_list[i]))
        log("NEW_BA-BA_OG={}".format(bas_diff_list[i]))
        log("\n")



