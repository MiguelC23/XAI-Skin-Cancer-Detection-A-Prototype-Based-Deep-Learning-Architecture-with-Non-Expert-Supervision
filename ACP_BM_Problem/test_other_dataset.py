import time
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import csv
from helpers import list_of_distances, make_one_hot

import os
import torch.nn.functional as F

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support



def _train_or_test(model, dataloader, optimizer=None, coefs=None, log=print,weights=None,
                   FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False,FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False):
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
    all_predicted, all_target = [], []
 

    for i, (image, label,ID,mask) in enumerate(dataloader):               
        input = image.cuda()
        target = label.cuda()
        mascaras=mask.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, upsampled_activation = model(input,mascaras)

            # compute loss
            #ISIC 2019 
            class_weights = torch.FloatTensor(weights).cuda()
            cross_entropy = torch.nn.functional.cross_entropy(output, target,weight=class_weights)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            all_predicted.extend(predicted.detach().cpu().tolist())
            all_target.extend(target.detach().cpu().tolist())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            
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

      

    
    """for a in range(num_classes):    
        log('\t{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted)[a])))"""
    
    #log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted))))
    
    
    if(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted,labels=[1, 2, 3,4,5,7]) ) ) )
        log('{0}'.format( classification_report(all_target, all_predicted,labels=[1, 2, 3,4,5,7]) ))
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted))))
        log('{0}'.format( classification_report(all_target, all_predicted)))
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==True):
        log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted,labels=[4,5]) ) ) )
        log('{0}'.format( classification_report(all_target, all_predicted,labels=[4,5])))

    
    if(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==False):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,labels=[1, 2, 3,4,5,7],
                                                        average='macro')
    elif(FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT==False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2==True):
        pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,labels=[4,5],
                                                        average='macro')
    accu=accuracy_score(all_target, all_predicted)
    

    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))

    
    if(model.module.num_classes==8):
        log('\n\tMalignant (1) vs Benign (0) information')
        BM_all_target = np.where(np.isin(all_target, [0,1, 4, 6]), 1, 0)
        BM_all_predicted = np.where(np.isin(all_predicted, [0,1, 4, 6]), 1, 0)
        for a in range(2):    
            log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target, BM_all_predicted)[a])))
        log('{0}'.format( classification_report(BM_all_target, BM_all_predicted) ))
            
    return accu,f1,rc


def test(model, dataloader, coefs=None, log=print, weights=None,
         FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False,FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False):
    log('\n\ttest')    
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs=coefs, log=log, weights=weights,
                          FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT,
                          FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2)



    



    
