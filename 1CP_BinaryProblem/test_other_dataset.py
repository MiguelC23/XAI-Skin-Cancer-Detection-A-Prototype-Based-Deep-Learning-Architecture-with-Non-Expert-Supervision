import time
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support


def _train_or_test(model, dataloader, optimizer=None,coefs=None, log=print,weights=None):
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
            output, min_distances, upsampled_activation, _ = model(input,mascaras)

            # compute loss
            #ISIC 2019 
            class_weights = torch.FloatTensor(weights).cuda()
            pos_weight = class_weights[1] / class_weights[0]  #Since melanoma is the label 0 and it assumes 1 is the positive class pos_weight mus be <1
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            cross_entropy = criterion(output, target.float())

            # evaluation statistics
            predicted = torch.round(torch.sigmoid(output)).long()
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


    
    log('{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted))))
    log('{0}'.format( classification_report(all_target, all_predicted)))
    pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,average='macro')
    accu=accuracy_score(all_target, all_predicted)
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))
            
    return accu,f1,rc


def test(model, dataloader, coefs=None, log=print,weights=None):
    log('\n\ttest')    
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,coefs=coefs, log=log,weights=weights)



    



    
