## 1-How to test the Black-Box model on the PH2 or Derm7pt test sets?
Simply run the file **Test_Black_Box.py**, and there is no need to provide arguments in the code line; just edit the following lines.

    PH2=False # True if you want to test on PH2 test dataset or false if you want to test on Derm7pt
    num_classes=8 # number of classes that the model has trained with. Choose 2 for the binary problem Melanoma vs Nevus and 8 for the multiclass problem.
    load_model_path =r"....pth" # Path to the model you want to test
    og_val_dir = r"..." Path to the folder containing the images of the ISIC 2019 validation set. Within this folder, 
    #there are 8 subfolders, each corresponding to a class: AK, BCC, BKL, DF, MEL, NV, SCC, VASC. 
    #Depending on whether the model was trained with 2 or 8 classes, the code only loads the necessary classes.
    
If PH2=True and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=True.
If PH2=False and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False.

    #This flags are important to show only results relate to the classes present in the Dataset we are testing
    #Because PH2 only has 2 classes and Derm7pt 6 classes
    #But if we trained the model with 8 classes need to pay attention to this flags
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True  #Always false if the model has been trained on the binary problem.
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False  #Always false if the model has been trained on the binary problem.

    if(PH2==True):
        test_dir=r"...." #Path to the 224x224x3 images of the PH2 test set.
        folder=r...\Test_Black-Box\ResultsPH2" #Path to the folder where we will save a text file containing the results of testing the model on the PH2 test set.
    else:
        test_dir=r"..." #Path to the 224x224x3 images of the Derm7pt test set.
        folder=r"...\Test_Black-Box\Results_Derm7pt" #Path to the folder where we will save a text file containing the results of testing the model on the Derm7pt test set.
    


The image folder of the PH2 dataset has two subfolders, one for each class (MEL and NV). In the case of the Derm7pt dataset, the image folder contains 6 subfolders, each corresponding to a class (BCC, BKL, DF, MEL, NV, VASC). It is worth noting that the images have dimensions of 224x224x3.
