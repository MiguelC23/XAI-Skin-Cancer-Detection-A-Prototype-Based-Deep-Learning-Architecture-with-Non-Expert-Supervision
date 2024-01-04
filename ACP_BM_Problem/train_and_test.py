import time
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from helpers import list_of_distances, make_one_hot
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import confusion_matrix, precision_recall_fscore_support

def _forbidding_loss(model, loss_images_loader,class_specific_penalization: bool):
    device = torch.device("cuda")
    conf_images, labels, names, idxs_patches = loss_images_loader

    distances = model.module.prototype_distances(conf_images) 
    n_imagens, L, _, _ = distances.shape
    distances_reshaped =  distances.view(n_imagens, L, -1)

    distances_escolhidas = torch.gather(distances_reshaped, 2, idxs_patches.view(-1, 1, 1).expand(-1, L, 1))
    distances_escolhidas=distances_escolhidas.squeeze(-1)

    sim = model.module.distance_2_similarity(distances_escolhidas)
    # shape [n_confound_images, n_prototypes]

    if class_specific_penalization:
        prototypes_of_confound_class = torch.t(
            model.module.prototype_class_identity[:, labels])
    else:
        # penalize all prototypes
        prototypes_of_confound_class = torch.ones_like(sim, device=device)

    prototypes_of_confound_class = prototypes_of_confound_class.to(device)
    l_dbg = torch.mean(torch.sum(sim * prototypes_of_confound_class, dim=1))
        
    del distances, distances_reshaped, distances_escolhidas, sim

    return l_dbg

def _remembering_loss_exp(model, positive_loss_images_loader):
    device = torch.device("cuda")
    pos_images, labels, names, idxs_patches = positive_loss_images_loader

    distances = model.module.prototype_distances(pos_images) 
    n_imagens, L, _, _ = distances.shape
    distances_reshaped =  distances.view(n_imagens, L, -1)

    distances_escolhidas = torch.gather(distances_reshaped, 2, idxs_patches.view(-1, 1, 1).expand(-1, L, 1))
    distances_escolhidas=distances_escolhidas.squeeze(-1)
    sim = model.module.distance_2_similarity(distances_escolhidas)
    prototypes_to_remember = torch.t(model.module.prototype_class_identity.to(device)[:, labels])
    
    l_dbg = torch.mean(torch.sum(sim * prototypes_to_remember, dim=1))
    
    del distances, distances_reshaped, distances_escolhidas, sim

    return l_dbg 

def _train_or_test(model, dataloader, optimizer=None, use_l1_mask=True, coefs=None, log=print, wandb=print, train_indication=False, loss_loader=None, positive_loss_loader=None,weights=None):
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
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_LM_cost = 0
    with_LM = False 
    total_LF_cost = 0
    total_LR_cost = 0

    all_predicted, all_target = [], []

 
    from settings import class_specific_penalization

    for i, (image, label,ID,mask) in enumerate(dataloader):
        # get one batch from finer datatloader
        if train_indication==True:
            with_LM = True
            LM_annotation = mask
            LM_annotation = LM_annotation.cuda()
        else:
            with_LM = False
                
        input = image.cuda()
        target = label.cuda()
        mascaras=mask.cuda()

        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances, upsampled_activation = model(input,mascaras)
            class_weights = torch.FloatTensor(weights).cuda()
            cross_entropy = torch.nn.functional.cross_entropy(output, target,weight=class_weights)


            
            max_dist = (model.module.prototype_shape[1]
                        * model.module.prototype_shape[2]
                        * model.module.prototype_shape[3])
            
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1) # torch.Size([batch_size]) 
            cluster_cost = torch.mean(max_dist - inverted_distances)

            
            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)


            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            if use_l1_mask:
                l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            else:
                l1 = model.module.last_layer.weight.norm(p=1) 
            
            # Mask loss LM
            LM_cost = 0
            if with_LM and (coefs['LM']>0):
                proto_num_per_class = model.module.num_prototypes // model.module.num_classes
                for index in range(image.shape[0]):
                        LM_cost += torch.norm(
                            upsampled_activation[index, label[index] * proto_num_per_class : (label[index] + 1) * proto_num_per_class]
                            * (1 * LM_annotation[index])
                        )


            forget_cost=0
            remember_cost=0
            if (coefs['LF']>0):
                if(loss_loader is not None):
                    forget_cost = _forbidding_loss(model=model,loss_images_loader=loss_loader,class_specific_penalization=class_specific_penalization)
            
            if (coefs['LR']>0):
                if(positive_loss_loader is not None):
                    remember_cost = _remembering_loss_exp(model=model, positive_loss_images_loader=positive_loss_loader)               

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            all_predicted.extend(predicted.detach().cpu().tolist())
            all_target.extend(target.detach().cpu().tolist())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_LM_cost += LM_cost
            total_avg_separation_cost += avg_separation_cost.item()
            total_LF_cost += forget_cost
            total_LR_cost += remember_cost

            
            total_all_cost+= (coefs['crs_ent'] * cross_entropy
                      + coefs['clst'] * cluster_cost
                      + coefs['sep'] * separation_cost
                      + coefs['l1'] * l1
                      + coefs['LM'] * LM_cost
                      +coefs['LF']*forget_cost
                      -coefs['LR']*remember_cost)

        # compute gradient and do SGD step
        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy
                      + coefs['clst'] * cluster_cost
                      + coefs['sep'] * separation_cost
                      + coefs['l1'] * l1
                      + coefs['LM'] * LM_cost
                      +coefs['LF']*forget_cost
                      -coefs['LR']*remember_cost)


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
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    log('\ttotal LM cost:\t{0}'.format(total_LM_cost / n_batches))
    log('\ttotal LF cost:\t{0}'.format(total_LF_cost / n_batches))
    log('\ttotal LR cost:\t{0}'.format(total_LR_cost / n_batches))
    log('\ttotal cost:\t{0}'.format(total_all_cost / n_batches))
    
    
    if(wandb!=print):
        if(train_indication==True):
            
            wandb.log({"cross ent train":total_cross_entropy / n_batches})
            wandb.log({"cluster train ":total_cluster_cost / n_batches})
            wandb.log({"separation train":total_separation_cost / n_batches})
            wandb.log({"total LM cost train":total_LM_cost / n_batches})
            wandb.log({"total LF cost train ":total_LF_cost / n_batches})
            wandb.log({"total LR cost train ":total_LR_cost / n_batches})
            wandb.log({"total cost train ":total_all_cost / n_batches})
        
        if(train_indication==False): #test
            
            wandb.log({"cross ent test":total_cross_entropy / n_batches})
            wandb.log({"cluster test":total_cluster_cost / n_batches})
            wandb.log({"separation test":total_separation_cost / n_batches})
            wandb.log({"total LM cost test":total_LM_cost / n_batches})
            wandb.log({"total LF cost test ":total_LF_cost / n_batches})
            wandb.log({"total LR cost test ":total_LR_cost / n_batches})
            wandb.log({"total cost test ":total_all_cost / n_batches})
    


    
    log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    if(wandb!=print):
        if(train_indication==True):
            wandb.log({"avg separation train":total_avg_separation_cost / n_batches})
        
        if(train_indication==False):
            wandb.log({"avg separation test":total_avg_separation_cost / n_batches})        




    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    
    if(wandb!=print):
        if(train_indication==True):
            wandb.log({"l1 train":model.module.last_layer.weight.norm(p=1).item()})
            wandb.log({"p dist pair train":p_avg_pair_dist.item()})
        
        if(train_indication==False):
            wandb.log({"l1 test":model.module.last_layer.weight.norm(p=1).item()})
            wandb.log({"p dist pair test":p_avg_pair_dist.item()})


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
    
    if(model.module.num_classes==8):
        log('\n\tMalignant (1) vs Benign (0) information')
        BM_all_target = np.where(np.isin(all_target, [0,1, 4, 6]), 1, 0)
        BM_all_predicted = np.where(np.isin(all_predicted, [0,1, 4, 6]), 1, 0)
        for a in range(2):    
            log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target, BM_all_predicted)[a])))
        log('{0}'.format( classification_report(BM_all_target, BM_all_predicted) ))
            
    return accu,f1,rc


def train(model, dataloader, optimizer, coefs=None, log=print, wandb=print, loss_loader=None, positive_loss_loader=None, weights=None, push_iterations=False):
    assert(optimizer is not None)
    
    log('\n\ttrain')
    if(push_iterations==True):
        log('\n\tLatent Space Before Prototype Layer Unchanged')
        model.eval() # beacuse of BatchNorm Layers in features
    else:
        model.train()
    
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs, log=log, wandb=wandb, train_indication=True, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)


def test(model, dataloader, coefs=None, log=print,wandb=print, loss_loader=None, positive_loss_loader=None,weights=None):
    log('\n\ttest')    
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs=coefs, log=log,wandb=wandb, train_indication=False, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader,weights=weights)


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
    
