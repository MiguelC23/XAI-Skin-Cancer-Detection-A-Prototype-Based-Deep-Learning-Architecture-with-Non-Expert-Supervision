import time
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, \
    precision_recall_fscore_support
from itertools import combinations
from LICD_and_VICN import LICD

def compute_distances(tensor1, tensor2):
    nimages, latent_space1, _, _ = tensor1.shape
    nprototypes, latent_space2, _, _ = tensor2.shape
    
    tensor1 = tensor1.view(nimages, latent_space1)
    tensor2 = tensor2.view(nprototypes, latent_space2)
    
    distances = torch.cdist(tensor1, tensor2)
    return distances

   
def _train_or_test(model, dataloader, optimizer=None,coefs=None, log=print,wandb=print,train_indication=False,weights=None):
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
    total_LICD_cost=0
    total_LPAMCS_cost=0
    

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
            output, min_distances, upsampled_activation,input_melanoma = model(input,mascaras)
            # compute loss
            #ISIC 2019
            class_weights = torch.FloatTensor(weights).cuda()
            pos_weight=class_weights[1] / class_weights[0] #Since melanoma is the label 0 and it assumes 1 is the positive class pos_weight mus be <1
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            cross_entropy = criterion(output, target.float())

            lpamcs_value=0
            if(coefs['LPAMCS']!=0):
                only_malignant_number=int(model.module.num_prototypes/2)
                numbers = list(range(only_malignant_number))
                pairs = list(combinations(numbers, 2))
                for index in range(image.shape[0]):                    
                    activations_maps_all_prototypes_for_this_image=upsampled_activation[index, 0:only_malignant_number]
                    max_value=torch.max(activations_maps_all_prototypes_for_this_image)
                    normalized_activation_maps = activations_maps_all_prototypes_for_this_image / max_value
                    pamcs_cost=0
                    for pair in pairs:
                        map_one=normalized_activation_maps[pair[0]]#/torch.max(normalized_activation_maps[pair[0]])
                        map_two=normalized_activation_maps[pair[1]]#/torch.max(normalized_activation_maps[pair[1]])
                        # Flatten the tensors
                        map_one_1D = map_one.view(-1)
                        map_two_1D = map_two.view(-1)
                        pamcs_cost += F.cosine_similarity(map_one_1D, map_two_1D, dim=0)+1                            
                    pamcs_cost=pamcs_cost/len(pairs)
                    lpamcs_value+=pamcs_cost                
                lpamcs_value=lpamcs_value/image.shape[0]
            
            licd_cost=LICD(model.module.prototype_vectors,model.module.num_classes)

            
            predicted = torch.round(torch.sigmoid(output)).long()
            all_predicted.extend(predicted.detach().cpu().tolist())
            all_target.extend(target.detach().cpu().tolist())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_LICD_cost += licd_cost
            total_LPAMCS_cost+=lpamcs_value
            
            

            total_all_cost+= (coefs['crs_ent'] * cross_entropy
                      +coefs['LICD']*licd_cost 
                      +coefs['LPAMCS']*lpamcs_value)

        # compute gradient and do SGD step
        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy
                      +coefs['LICD']*licd_cost   
                      +coefs['LPAMCS']*lpamcs_value)


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
    log('\ttotal LICD cost :\t{0}'.format(total_LICD_cost / n_batches))
    log('\ttotal LPAMCS cost :\t{0}'.format(total_LPAMCS_cost / n_batches))        
    log('\ttotal cost:\t{0}'.format(total_all_cost / n_batches))
    
    if(wandb!=print):
        if(train_indication==True):#train
            
            wandb.log({"cross ent train":total_cross_entropy / n_batches})
            wandb.log({"total LICD cost train":total_LICD_cost / n_batches})
            wandb.log({"total LPAMCS cost train":total_LPAMCS_cost / n_batches})             
            wandb.log({"total cost train ":total_all_cost / n_batches})
        
        if(train_indication==False): #test
            
            wandb.log({"cross ent test":total_cross_entropy / n_batches})
            wandb.log({"total LICD cost test":total_LICD_cost / n_batches})
            wandb.log({"total LPAMCS cost test":total_LPAMCS_cost / n_batches})                      
            wandb.log({"total cost test ":total_all_cost / n_batches})
    

    for a in range(model.module.num_classes):    
        log('\t{0}'.format( np.array2string(confusion_matrix(all_target, all_predicted)[a])))

    log('{0}'.format( classification_report(all_target, all_predicted) ))

    pr, rc, f1, sp = precision_recall_fscore_support(all_target, all_predicted,
                                                    average='macro')
    accu=accuracy_score(all_target, all_predicted)
    
    log('\tAccuracy : \t{0}'.format(accu))
    log('\tmacro-averaged precision : \t{0}'.format(pr))
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc))
    log('\tmacro-averaged F1 score: \t{0}'.format(f1))

    if(wandb!=print):
        if(train_indication==True):
            wandb.log({"Accuracy train":accu})
            wandb.log({"macro-averaged precision train":pr})
            wandb.log({"BA train":rc})
            wandb.log({"macro-averaged F1 score train":f1})


        if(train_indication==False):
            wandb.log({"Accuracy test":accu})
            wandb.log({"macro-averaged precision test":pr})
            wandb.log({"BA test":rc})
            wandb.log({"macro-averaged F1 score test":f1})
                
    return accu,f1,rc


def train(model, dataloader, optimizer, coefs=None, log=print,wandb=print,weights=None,push_iterations=False):
    assert(optimizer is not None)
    
    log('\n\ttrain')
    if(push_iterations==True):
        log('\n\tLatent Space Before Prototype Layer Unchanged')
        model.eval() # beacuse of BatchNorm Layers in features
    else:
        model.train()
    
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs, log=log, wandb=wandb, train_indication=True, weights=weights)


def test(model, dataloader, coefs=None, log=print,wandb=print,weights=None):
    log('\n\ttest')    
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,coefs=coefs, log=log,wandb=wandb,train_indication=False,weights=weights)


def last_only(model, log=print,wandb=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')
    
def warm_only(model, log=print,wandb=print,Fixed_prototypes_during_training_initialized_orthogonally=False):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = not Fixed_prototypes_during_training_initialized_orthogonally
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')

def joint(model, log=print,wandb=print,Fixed_prototypes_during_training_initialized_orthogonally=False):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = not Fixed_prototypes_during_training_initialized_orthogonally
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
    
