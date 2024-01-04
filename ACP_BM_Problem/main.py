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
    import torchvision.transforms as transforms
    import re
    from helpers import makedir
    import model
    import push
    import train_and_test as tnt
    import save
    from log import create_logger
    from preprocess import mean, std, preprocess_input_function
    import random
    import wandb
    from Dataset import ISIC2019_Dataset
    from settings import random_seed_number
    from settings import img_size, prototype_shape, num_classes, prototype_activation_function, add_on_layers_type, prototype_activation_function_in_numpy
    from settings import base_architecture,last_layer_weight,topk_k,latent_shape,experiment_run
    from settings import folder_path_to_save_runs
    from settings import train_dir, test_dir, train_push_dir,train_batch_size, test_batch_size, train_push_batch_size
    from settings import train_mask_dir,online_augmentation
    from settings import LP_MASKED,Fixed_prototypes_during_training_initialized_orthogonally
    from settings import load_model_path
    from settings import joint_optimizer_lrs, joint_lr_step_size
    from settings import warm_optimizer_lrs
    from settings import last_layer_optimizer_lr
    from settings import coefs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs,diversity,number_iterations,class_specific_penalization,Test_results_with_test_masks
    from settings import number_of_prototypes_per_class

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
       
    # book keeping namings and code
    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
    prototype_shape = (prototype_shape[0], latent_shape, prototype_shape[2], prototype_shape[3])
    print("Protoype shape: ", prototype_shape)
    model_dir = folder_path_to_save_runs  +'/' + base_architecture + '/' + experiment_run + '/'
    print("saving models to: ", model_dir)
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # all datasets
    # train set
    train_dataset = ISIC2019_Dataset(train_dir, train_mask_dir, is_train=True,number_classes=num_classes,augmentation=online_augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    # push set
    train_push_dataset = ISIC2019_Dataset(train_push_dir, train_mask_dir, is_train=True,is_push=True,number_classes=num_classes)
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    # test set
    from settings import test_mask_dir
    test_dataset = ISIC2019_Dataset(test_dir, mask_dir=test_mask_dir, is_train=False,number_classes=num_classes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    

    
    from sklearn.utils.class_weight import compute_class_weight
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
        
    loss_loader=None
    positive_loss_loader=None
    if(coefs['LF']!=0 or coefs['LR']!=0):
        from settings import forbidden_protos_directory,remembering_protos_directory
        from DatatsetDebug import ImageFolderLoss  

        loss_data = ImageFolderLoss(forbidden_protos_directory,torch.device("cuda"))
        loss_loader=loss_data.get_all() if loss_data is not None else None
        
        positive_loss_data = ImageFolderLoss(remembering_protos_directory,torch.device("cuda"))
        positive_loss_loader=positive_loss_data.get_all() if positive_loss_data is not None else None

        log('forbidden_protos_directory: {0}'.format(forbidden_protos_directory))
        log('remembering_protos_directory: {0}'.format(remembering_protos_directory))
    
    
    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set location: {0}'.format(train_dir))
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('training masks location: {0}'.format(train_mask_dir))
    log('push set location: {0}'.format(train_push_dir))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set location: {0}'.format(test_dir))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))
    if(base_architecture!='vgg16'):
        log("Considering 7x7 grid Using topk_k coeff from bash args: {0}, which is {1:.4}%".format(topk_k, float(topk_k)*100./(7*7))) # for prototype size 1x1 on 7x7 grid experminents
    else:
        log("Considering 14x14 grid Using topk_k coeff from bash args: {0}, which is {1:.4}%".format(topk_k, float(topk_k)*100./(14*14))) # for prototype size 1x1 on 7x7 grid experminents
    

    
    if load_model_path is not None: #Continue training without debug
        ppnet = torch.load(load_model_path)
        log('starting from model: {0}'.format(load_model_path))
    elif load_model_path==None:#Training from Scratch
        ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    topk_k=topk_k,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type,
                                    last_layer_weight=last_layer_weight,LP_MASKED=LP_MASKED,
                                    Fixed_prototypes_during_training_initialized_orthogonally=Fixed_prototypes_during_training_initialized_orthogonally)
    
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    #define optimizer
    if(ppnet.Fixed_prototypes_during_training_initialized_orthogonally==True):
        joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, 
        {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3}]
    else:
        joint_optimizer_specs = \
        [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, 
        {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
        ]

    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    if(ppnet.Fixed_prototypes_during_training_initialized_orthogonally==True):
        warm_optimizer_specs = [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3}]
    else:
        warm_optimizer_specs = \
        [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
        ]

    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    
    print('Parameters of last layer:\n', ppnet.last_layer.weight,'\n')
    
    # weighting of different training losses
    log('Loss function coefficients:\n\tcrs_ent={};\n\tclst={};\n\tsep={};\n\tl1={};\n\tLM={};\n\tLF={};\n\tLR={};\n'.format(coefs['crs_ent'],coefs['clst'],coefs['sep'],coefs['l1'],coefs['LM'],coefs['LF'],coefs['LR']))

    if(num_classes==2):
        project_name="ACP_Binary_Problem" # Prototypes for all calsses and binary problem melanoma vs nevus.
    elif (num_classes==8):
        project_name="ACP_Multiclass_Problem" # Prototypes for all calsses and multiclass problem with 8 classes AK BCC BKL DF MEL NV SCC VASC

    wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,
    
    # track hyperparameters and run metadata
    config={
    "base_architecture": base_architecture,
    "img_size": img_size,
    "prototype_shape": prototype_shape,
    "num_classes" :num_classes,
    "latent_shape":latent_shape,
    "last_layer_weight":last_layer_weight,
    "topk_k":topk_k,
    "random_seed_number":random_seed_number,
    "prototype_activation_function":prototype_activation_function,
    "prototype_activation_function_in_numpy":prototype_activation_function_in_numpy,
    "add_on_layers_type":add_on_layers_type,
    "experiment_run":experiment_run,
    "train_batch_size":train_batch_size,
    "test_batch_size":test_batch_size,
    "train_push_batch_size":train_push_batch_size,
    "joint_optimizer_lrs":joint_optimizer_lrs,
    "joint_lr_step_size":joint_lr_step_size,
    "warm_optimizer_lrs":warm_optimizer_lrs,
    "last_layer_optimizer_lr":last_layer_optimizer_lr,
    "coefs":coefs,
    "num_train_epochs":num_train_epochs,
    "num_warm_epochs":num_warm_epochs,
    "push_start":push_start,
    "push_epochs":push_epochs,
    "diversity":diversity,
    "Number_Prototypes":prototype_shape[0],
    "PC":"Miguel",
    "number_iterations":number_iterations,
    "class_specific_penalization":class_specific_penalization,
    "LP_MASKED":LP_MASKED,
    "number_of_prototypes_per_class":number_of_prototypes_per_class,
    "Fixed_prototypes_during_training_initialized_orthogonally":Fixed_prototypes_during_training_initialized_orthogonally,
    "Test_results_with_test_masks":Test_results_with_test_masks,
    "OA":online_augmentation}
    )
    
    wandb.run.name=experiment_run

     # train the model
    log('start training')
    
    import copy

    #train_auc = []
    #test_auc = []
    train_accu=[]
    train_F1_score=[]
    train_ba=[]

    test_accu=[]
    test_F1_score=[]
    test_ba=[]

    currbest, best_epoch = 0, -1
    currbest_F1,best_epoch_F1=0, -1
    currbest_ba,best_epoch_ba=0,-1

    best_ACCs_considering_only_push=[]
    best_ACCs_considering_only_push_iteration=[]
       

    best_F1s_considering_only_push=[]
    best_F1s_considering_only_push_iteration=[]
       

    best_BAs_considering_only_push=[]
    best_BAs_considering_only_push_iteration=[]

    all_push_epochs=[]

    push_with_no_iterations_ACC=[]
    push_with_no_iterations_F1=[]
    push_with_no_iterations_BA=[]
    push_with_no_iterations_epochs=[]
        
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))
        
        if epoch < num_warm_epochs:
            
            tnt.warm_only(model=ppnet_multi, log=log,wandb=wandb, Fixed_prototypes_during_training_initialized_orthogonally=ppnet.Fixed_prototypes_during_training_initialized_orthogonally)
            accu,F1_score,ba = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, coefs=coefs, log=log, wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)
        else:
            tnt.joint(model=ppnet_multi, log=log,wandb=wandb, Fixed_prototypes_during_training_initialized_orthogonally=ppnet.Fixed_prototypes_during_training_initialized_orthogonally)
            accu,F1_score,ba = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, coefs=coefs, log=log,wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)
            joint_lr_scheduler.step()

        accu_test,F1_score_test,ba_test = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs, log=log, wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)

        train_accu.append(accu)
        train_F1_score.append(F1_score)
        train_ba.append(ba)

        if currbest < accu_test:
            currbest = accu_test
            best_epoch = epoch

        if currbest_F1 < F1_score_test:
            currbest_F1  = F1_score_test
            best_epoch_F1 = epoch

        if currbest_ba < ba_test:
            currbest_ba  = ba_test
            best_epoch_ba = epoch            

        log("\n\tcurrent best accuracy is: \t\t{} at epoch {}".format(currbest, best_epoch))
        log("\tcurrent best F1 is: \t\t{} at epoch {}".format(currbest_F1, best_epoch_F1))
        log("\tcurrent best BA is: \t\t{} at epoch {}".format(currbest_ba, best_epoch_ba))
        log("\tParameters of last layer weights:\n{}\n".format( ppnet.last_layer.weight))
        

        test_accu.append(accu_test)
        test_F1_score.append(F1_score_test)
        test_ba.append(ba_test)

        plt.plot(train_accu, "b", label="train")
        plt.plot(test_accu, "r", label="test")
        plt.legend()
        plt.savefig(model_dir + 'train_test_accu.png')
        plt.close()

        plt.plot(train_F1_score, "b", label="train")
        plt.plot(test_F1_score, "r", label="test")
        plt.legend()
        plt.savefig(model_dir + 'train_test_F1.png')
        plt.close()

        plt.plot(train_ba, "b", label="train")
        plt.plot(test_ba, "r", label="test")
        plt.legend()
        plt.savefig(model_dir + 'train_test_ba.png')
        plt.close()

        if epoch >= push_start and epoch in push_epochs:
            
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=True,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
                prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                wandb=wandb,
                diversity=diversity)
            
            accu_test,F1_score_test,ba_test = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs, log=log, wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push',ba=ba_test,target_ba=0,log=log)
            
            push_with_no_iterations_ACC.append(accu_test)
            push_with_no_iterations_F1.append(F1_score_test)
            push_with_no_iterations_BA.append(ba_test)
            push_with_no_iterations_epochs.append(epoch)

            if prototype_activation_function != 'linear':

                tnt.last_only(model=ppnet_multi, log=log,wandb=wandb)
                
                currbest_F1_iteration,best_Fiteration=0,-1
                currbest_acc_iteration,best_aiteration=0,-1
                currbest_ba_iteration,best_ba_iteration=0,-1
                
                for i in range(number_iterations):
                    log('iteration: \t{0}'.format(i))
                    
                    accu,F1_score,ba = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, coefs=coefs, log=log, wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights, push_iterations=True)
                    
                    accu_test,F1_score_test,ba_test = tnt.test(model=ppnet_multi, dataloader=test_loader, coefs=coefs, log=log, wandb=wandb, loss_loader=loss_loader, positive_loss_loader=positive_loss_loader, weights=weights)
                    
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', ba=ba_test, target_ba= currbest_ba_iteration, log=log)
                    
                    train_accu.append(accu)
                    train_F1_score.append(F1_score)
                    train_ba.append(ba)

                    if currbest < accu_test:
                        currbest = accu_test
                        best_epoch = epoch

                    if currbest_F1 < F1_score_test:
                        currbest_F1  = F1_score_test
                        best_epoch_F1 = epoch

                    if currbest_ba < ba_test:
                        currbest_ba  = ba_test
                        best_epoch_ba = epoch  

                    if currbest_acc_iteration < accu_test:
                        currbest_acc_iteration = accu_test
                        best_aiteration = i

                    if currbest_F1_iteration< F1_score_test:
                        currbest_F1_iteration  = F1_score_test
                        best_Fiteration = i

                    if currbest_ba_iteration< ba_test:
                        currbest_ba_iteration  = ba_test
                        best_ba_iteration = i

                    

                    log("\n\tcurrent best accuracy is: \t\t{} at epoch {}".format(currbest, best_epoch))
                    log("\tcurrent best F1 is: \t\t{} at epoch {}".format(currbest_F1, best_epoch_F1))
                    log("\tcurrent best BA is: \t\t{} at epoch {}".format(currbest_ba, best_epoch_ba))

                    log("\tParameters of last layer weights:\n{}\n".format( ppnet.last_layer.weight))
                    

                    log("\tFor push at epoch {} current best accuracy is: \t\t{} at iteration {}".format(epoch,currbest_acc_iteration,best_aiteration))
                    log("\tFor push at epoch {} current best F1 is: \t\t{} at iteration {}".format(epoch,currbest_F1_iteration,best_Fiteration))
                    log("\tFor push at epoch {} current best BA is: \t\t{} at iteration {}".format(epoch,currbest_ba_iteration,best_ba_iteration))
                    

                    test_accu.append(accu_test)
                    test_F1_score.append(F1_score_test)
                    test_ba.append(ba_test)

                    plt.plot(train_accu, "b", label="train")
                    plt.plot(test_accu, "r", label="test")
                    plt.legend()
                    plt.savefig(model_dir + 'train_test_accu.png')
                    plt.close()

                    plt.plot(train_F1_score, "b", label="train")
                    plt.plot(test_F1_score, "r", label="test")
                    plt.legend()
                    plt.savefig(model_dir + 'train_test_F1.png')
                    plt.close()
                    
                    plt.plot(train_ba, "b", label="train")
                    plt.plot(test_ba, "r", label="test")
                    plt.legend()
                    plt.savefig(model_dir + 'train_test_ba.png')
                    plt.close()

                best_ACCs_considering_only_push.append(currbest_acc_iteration)
                best_ACCs_considering_only_push_iteration.append(best_aiteration)
                

                best_F1s_considering_only_push.append(currbest_F1_iteration)
                best_F1s_considering_only_push_iteration.append(best_Fiteration)
                

                best_BAs_considering_only_push.append(currbest_ba_iteration)
                best_BAs_considering_only_push_iteration.append(best_ba_iteration)

                all_push_epochs.append(epoch)

    if(number_iterations==0):
        max_push_with_no_iterations_ACC=max(push_with_no_iterations_ACC)
        max_push_with_no_iterations_F1=max(push_with_no_iterations_F1)
        max_push_with_no_iterations_BA=max(push_with_no_iterations_BA)

        index1_no_i=push_with_no_iterations_ACC.index(max_push_with_no_iterations_ACC)
        index2_no_i=push_with_no_iterations_F1.index(max_push_with_no_iterations_F1)
        index3_no_i=push_with_no_iterations_BA.index(max_push_with_no_iterations_BA)

        epoch_1_no_i=push_with_no_iterations_epochs[index1_no_i]
        epoch_2_no_i=push_with_no_iterations_epochs[index2_no_i]
        epoch_3_no_i=push_with_no_iterations_epochs[index3_no_i]

        log("\tWe are only taking into account the outcomes of push operations that don't involve any iterations.")
        log("\t\tThe Best ACC is {} from epoch {}".format(max_push_with_no_iterations_ACC,epoch_1_no_i))
        log("\t\tThe Best F1 is {}  from epoch {}".format(max_push_with_no_iterations_F1,epoch_2_no_i))
        log("\t\tThe Best BA is {}  from epoch {}".format(max_push_with_no_iterations_BA,epoch_3_no_i))

        wandb.log({"BEST_ACC_PUSH":max_push_with_no_iterations_ACC,"BEST_ACC_PUSH_EPOCH":epoch_1_no_i,"BEST_ACC_PUSH_EPOCH_ITERATION":-1})
        wandb.log({"BEST_F1_PUSH":max_push_with_no_iterations_F1,"BEST_F1_PUSH_EPOCH":epoch_2_no_i,"BEST_F1_PUSH_EPOCH_ITERATION":-1})
        wandb.log({"BEST_BA_PUSH":max_push_with_no_iterations_BA,"BEST_BA_PUSH_EPOCH":epoch_3_no_i,"BEST_BA_PUSH_EPOCH_ITERATION":-1})
    else:
        best_acc_of_all_push=max(best_ACCs_considering_only_push)
        index1=best_ACCs_considering_only_push.index(best_acc_of_all_push)
        best_acc_of_all_push_iteration=best_ACCs_considering_only_push_iteration[index1]
        best_acc_of_all_push_epoch=all_push_epochs[index1]

        best_F1_of_all_push=max(best_F1s_considering_only_push)
        index2=best_F1s_considering_only_push.index(best_F1_of_all_push)
        best_F1_of_all_push_iteration=best_F1s_considering_only_push_iteration[index2]
        best_F1_of_all_push_epoch=all_push_epochs[index2]

        best_BA_of_all_push=max(best_BAs_considering_only_push)
        index3=best_BAs_considering_only_push.index(best_BA_of_all_push)
        best_BA_of_all_push_iteration=best_BAs_considering_only_push_iteration[index3]
        best_BA_of_all_push_epoch=all_push_epochs[index3]
        
        log("\n\tConsidering only results from push:")
        log("\t\tThe Best ACC is {} at iteration {} from epoch {}".format(best_acc_of_all_push,best_acc_of_all_push_iteration,best_acc_of_all_push_epoch))
        log("\t\tThe Best F1 is {} at iteration {} from epoch {}".format(best_F1_of_all_push,best_F1_of_all_push_iteration,best_F1_of_all_push_epoch))
        log("\t\tThe Best BA is {} at iteration {} from epoch {}".format(best_BA_of_all_push,best_BA_of_all_push_iteration,best_BA_of_all_push_epoch))

        wandb.log({"BEST_ACC_PUSH":best_acc_of_all_push,"BEST_ACC_PUSH_EPOCH":best_acc_of_all_push_epoch,"BEST_ACC_PUSH_EPOCH_ITERATION":best_acc_of_all_push_iteration})
        wandb.log({"BEST_F1_PUSH":best_F1_of_all_push,"BEST_F1_PUSH_EPOCH":best_F1_of_all_push_epoch,"BEST_F1_PUSH_EPOCH_ITERATION":best_F1_of_all_push_iteration})
        wandb.log({"BEST_BA_PUSH":best_BA_of_all_push,"BEST_BA_PUSH_EPOCH":best_BA_of_all_push_epoch,"BEST_BA_PUSH_EPOCH_ITERATION":best_BA_of_all_push_iteration})