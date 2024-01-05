if __name__ == '__main__':
    import os
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import torch
    import torch.utils.data
    from log import create_logger
    import random
    from Dataset import PH2_Dataset_or_Derm7pt,ISIC2019_Dataset
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    random_seed_number=4
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    
    PH2=False # True if you want to test on PH2 test dataset or false if you want to test on Derm7pt
    num_classes=8 # number f classes that the model has trained with
    load_model_path =r"E:\Coisas da Tese\Test_Black-Box\NC8\NC8_R18_5_0.6891.pth"
    og_val_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val"

    #This flags are important to show only results relate to the classes present in the Dataset we are testing
    #Because PH2 only has 2 classes and Derm7pt 6 classes
    #But if we trained the model with 8 classes need to pay attention to this flags
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True
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
        testing_log_name = f"{model_name}_{filename}_PH2.txt"
    else:
        testing_log_name = f"{model_name}_{filename}_Derm7pt.txt"


    log, logclose = create_logger(log_filename=os.path.join(folder, testing_log_name))
    log("\t\t\tModel {}\n".format(testing_log_name))
    log(load_model_path)
    if(PH2==True):
        log("\t\t\tAnalysis of the PH2 dataset\n")
    else:
        log("\t\t\tAnalysis of the derm7pt dataset\n")

    model = torch.load(load_model_path)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    log("\n========================Test Set Performance========================\n")
    test_batch_size = 100
    # test set
    test_dataset = PH2_Dataset_or_Derm7pt(test_dir, mask_dir=None, is_train=False,number_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    log('\t\t\ttest set size: {0}\n'.format(len(test_loader.dataset)))

    labels_test = [item[1] for item in test_dataset.ids]
    labels_test=np.array(labels_test)
    classes_test,counts= np.unique(labels_test,return_counts=True)
    log('\t\t\ttest labels: {0}\n'.format(classes_test))
    log('\t\t\ttest labels counts: {0}\n'.format(counts))

    all_predicted_val, all_target_val = [], []
    model.eval()
    n_batches_val=0
    #total_cross_entropy_val = 0
    device = torch.device("cuda")
    with torch.no_grad():
        for images, labels, lixo1, lixo2 in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)


            n_batches_val += 1
            _, predicted = torch.max(outputs.data, 1)
            all_predicted_val.extend(predicted.detach().cpu().tolist())
            all_target_val.extend(labels.detach().cpu().tolist())
    

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



    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))
    if(num_classes==8):
        log('\n\tMalignant (1) vs Benign (0) information')
        BM_all_target = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
        BM_all_predicted = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
        for a in range(2):    
            log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target, BM_all_predicted)[a])))
        log('{0}'.format( classification_report(BM_all_target, BM_all_predicted) ))

    #Validation original dataset
    log("\n========================Validation Set Performance========================\n")
    ogval_dataset = ISIC2019_Dataset(og_val_dir, mask_dir=None, is_train=False,number_classes=num_classes)
    ogval_loader = torch.utils.data.DataLoader(
        ogval_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    
    all_predicted_val, all_target_val = [], []
    model.eval()
    n_batches_val=0
    #total_cross_entropy_val = 0
    device = torch.device("cuda")
    with torch.no_grad():
        for images, labels, lixo1, lixo2 in ogval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)


            n_batches_val += 1
            _, predicted = torch.max(outputs.data, 1)
            all_predicted_val.extend(predicted.detach().cpu().tolist())
            all_target_val.extend(labels.detach().cpu().tolist())
    

    log('{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val))))

    log('{0}'.format( classification_report(all_target_val, all_predicted_val) ))

    pr_val, rc_val, f1_val, sp_val = precision_recall_fscore_support(all_target_val, all_predicted_val,average='macro')
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_val))

    if(num_classes==8):
        log('\n\tMalignant (1) vs Benign (0) information')
        BM_all_target = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
        BM_all_predicted = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
        for a in range(2):    
            log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target, BM_all_predicted)[a])))
        log('{0}'.format( classification_report(BM_all_target, BM_all_predicted) ))

