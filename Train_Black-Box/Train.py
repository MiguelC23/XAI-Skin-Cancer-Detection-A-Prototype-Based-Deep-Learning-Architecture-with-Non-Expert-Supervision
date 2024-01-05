if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.optim as optim
    from helpers import makedir
    import save
    from log import create_logger
    import random
    from Dataset import ISIC2019_Dataset
    from settings import random_seed_number
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics._classification import confusion_matrix, precision_recall_fscore_support
    from settings import train_dir, test_dir,train_batch_size, test_batch_size,num_classes,OA
    from sklearn.utils.class_weight import compute_class_weight
    from settings import base_architecture,load_model_path,load_model
    import torchvision.models as models
    from settings import experiment_run,num_train_epochs,folder_path_to_save_runs,lr

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    # all datasets
    # train set
    train_dataset = ISIC2019_Dataset(train_dir, mask_dir=None, is_train=True,number_classes=num_classes,augmentation=OA)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    # test set
    test_dataset = ISIC2019_Dataset(test_dir, mask_dir=None, is_train=False,number_classes=num_classes)
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



    class_weights=weights
    # Define model
    model_choice=base_architecture
    if(load_model==False):
      # Verificar a escolha do modelo e definir o modelo correspondente
      if model_choice == "resnet18":
          model = models.resnet18(pretrained=True)
          num_features = model.fc.in_features
          model.fc = nn.Linear(num_features, len(class_weights))
      elif model_choice == "resnet50":
          model = models.resnet50(pretrained=True)
          num_features = model.fc.in_features
          model.fc = nn.Linear(num_features, len(class_weights))
      elif model_choice == "densenet169":
          model = models.densenet169(pretrained=True)
          num_features = model.classifier.in_features
          model.classifier = nn.Linear(num_features, len(class_weights))
      elif model_choice == "eb3":
          model = models.efficientnet_b3(pretrained=True)
          num_features = model.classifier[-1].in_features
          model.classifier[-1] = nn.Linear(num_features, len(class_weights))
      elif model_choice == "vgg16":
          model = models.vgg16(pretrained=True)
          model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_weights))
      else:
          raise ValueError("Escolha de modelo inválida!")
    elif(load_model==True):
      model=torch.load(load_model_path)


    model = model.to(device)
    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Define loss function with weighted cross entropy
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Train the model
    model_dir = folder_path_to_save_runs + '/' + base_architecture + '/' + experiment_run + '/'
    print("saving models to: ", model_dir)
    makedir(model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    currbest_ba,best_epoch_ba=0,-1
    currbest, best_epoch = 0, -1
    currbest_F1,best_epoch_F1=0, -1

    train_accu=[]
    train_F1_score=[]
    train_ba=[]

    test_accu=[]
    test_F1_score=[]
    test_ba=[]

    train_loss=[]
    val_loss=[]

    for epoch in range(num_train_epochs):
        log('\n\tepoch: \t{0}'.format(epoch))
        # Train
        log('\ttrain')
        all_predicted_train, all_target_train = [], []
        model.train()
        n_batches_train=0
        total_cross_entropy_train = 0
        for images, labels, lixo1, lixo2 in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted_train.extend(predicted.detach().cpu().tolist())
            all_target_train.extend(labels.detach().cpu().tolist())



            loss = criterion(outputs, labels)
            n_batches_train += 1
            total_cross_entropy_train += loss.item()
            loss.backward()
            optimizer.step()
        
        loss_training=total_cross_entropy_train / n_batches_train
        log('\tcross ent train: \t{0}'.format(total_cross_entropy_train / n_batches_train))

        for a in range(num_classes):    
            log('\t{0}'.format( np.array2string(confusion_matrix(all_target_train, all_predicted_train)[a])))

        log('{0}'.format( classification_report(all_target_train, all_predicted_train) ))

        pr_train, rc_train, f1_train, sp_train = precision_recall_fscore_support(all_target_train, all_predicted_train,
                                                        average='macro')
        accu_train=accuracy_score(all_target_train, all_predicted_train)
        
        log('\tmacro-averaged precision : \t{0}'.format(pr_train))
        log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_train))
        log('\tmacro-averaged F1 score: \t{0}'.format(f1_train))
        log('\ttrain accuracy: \t{0}'.format(accu_train))

        if(num_classes==8):
          log('\n\tMalignant (1) vs Benign (0) information')
          BM_all_target_train = np.where(np.isin(all_target_train, [0,1, 4, 6]), 1, 0)
          BM_all_predicted_train = np.where(np.isin(all_predicted_train, [0,1, 4, 6]), 1, 0)
          for a in range(2):    
              log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target_train, BM_all_predicted_train)[a])))
          log('{0}'.format( classification_report(BM_all_target_train, BM_all_predicted_train) ))
        
        
        # Validate
        log('\n\tvalidation')
        all_predicted_val, all_target_val = [], []
        model.eval()
        n_batches_val=0
        total_cross_entropy_val = 0
        with torch.no_grad():
            for images, labels, lixo1, lixo2 in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                n_batches_val += 1
                total_cross_entropy_val += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                all_predicted_val.extend(predicted.detach().cpu().tolist())
                all_target_val.extend(labels.detach().cpu().tolist())
        
        loss_validation=total_cross_entropy_val / n_batches_val
        log('\tcross ent val: \t{0}'.format(total_cross_entropy_val / n_batches_val))

        for a in range(num_classes):    
            log('\t{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val)[a])))

        log('{0}'.format( classification_report(all_target_val, all_predicted_val) ))

        pr_val, rc_val, f1_val, sp_val = precision_recall_fscore_support(all_target_val, all_predicted_val,
                                                        average='macro')
        accu_val=accuracy_score(all_target_val, all_predicted_val)
        
        log('\tmacro-averaged precision : \t{0}'.format(pr_val))
        log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_val))
        log('\tmacro-averaged F1 score: \t{0}'.format(f1_val))
        log('\tvalidation accuracy: \t{0}'.format(accu_val))

        if(num_classes==8):
          log('\n\tMalignant (1) vs Benign (0) information')
          BM_all_target_val = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
          BM_all_predicted_val = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
          for a in range(2):    
              log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target_val, BM_all_predicted_val)[a])))
          log('{0}'.format( classification_report(BM_all_target_val, BM_all_predicted_val) ))

        save.save_model_w_condition(model=model, model_dir=model_dir, model_name=str(epoch),ba=rc_val,target_ba=currbest_ba,log=log)
        if currbest_ba < rc_val:
            currbest_ba  = rc_val
            best_epoch_ba = epoch

        if currbest < accu_val:
            currbest = accu_val
            best_epoch = epoch

        if currbest_F1 < f1_val:
            currbest_F1  = f1_val
            best_epoch_F1 = epoch
        
        log("\n\tcurrent best accuracy is: \t\t{} at epoch {}".format(currbest, best_epoch))
        log("\tcurrent best F1 is: \t\t{} at epoch {}".format(currbest_F1, best_epoch_F1))
        log("\tcurrent best BA is: \t\t{} at epoch {}".format(currbest_ba, best_epoch_ba))
        

        train_accu.append(accu_train)
        train_F1_score.append(f1_train)
        train_ba.append(rc_train)

        test_accu.append(accu_val)
        test_F1_score.append(f1_val)
        test_ba.append(rc_val)

        train_loss.append(loss_training)
        val_loss.append(loss_validation)

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


        plt.plot(train_loss, "b", label="train")
        plt.plot(val_loss, "r", label="test")
        plt.legend()
        plt.savefig(model_dir + 'train_test_loss.png')
        plt.close()      

        
