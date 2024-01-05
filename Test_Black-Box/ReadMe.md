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

## 2- Conduct the evaluation of the following XAI metrics: G3-Truthfulness and G4-Informative Plausibility on the Black-box model.
Just run the following file **XAI_MG3_MG4_Black_Box.py**; no need to provide arguments in the terminal command line, but you should edit the following lines as necessary:

    PH2=True # Set to True to perform evaluation on PH2 or False to perform evaluation on Derm7pt
    test_masks_flag=True # Put this at True since we need the masks to perform the evaluation of G3 and G4.
    num_classes=2 #  number of classes that the model has trained with. Choose 2 for the binary problem Melanoma vs Nevus and 8 for the multiclass problem.
    load_model_path =r"....pth" #path to the model you want to evaluate
    random_heatmaps_flag=False # When true, perform the evaluation based on random heatmaps rather than the true model heatmaps, used for comparative purposes.

If PH2=True and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=True.
If PH2=False and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False.

    #This flags are important to show only results relate to the classes present in the Dataset we are testing
    #Because PH2 only has 2 classes and Derm7pt 6 classes
    #But if we trained the model with 8 classes need to pay attention to this flags
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False  # Always false if the model has been trained on the binary problem.
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False #Always false if the model has been trained on the binary problem.

    if(PH2==True):
        test_dir=r"...Path to the folder containing the MEL and NV folders of the test images from the PH2 dataset..."
        folder=r"...\Test_Black-Box\ResultsPH2" #Path to the folder where a text file with the results will be saved.
    else:
        test_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\derm7pt_like_ISIC2019\train_val_test_224"
        folder=r"...\Test_Black-Box\Results_Derm7pt" Path to the folder where a text file with the results will be saved.


    if(test_masks_flag==True):
        if(PH2==True):
                test_mask_dir=r"..." #Path to the folder where two folders are located, MEL and NV, i.e., the binary masks where 1 represents the non-relevant and 0 represents the relevant.
                        #One binary mask per image. 
                        #In other words, the masks are inverted. For the PH2 test dataset.
        elif(PH2==False):
                test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\PH2_DERM7PT\DERM7PT_FINE_MASKS_224" #Path to the folder where six folders are located,(BCC, BKL, DF, MEL, NV, VASC), 
                #i.e., the binary masks where 1 represents the non-relevant and 0 represents the relevant. #One binary mask per image. 
                #In other words, the masks are inverted. For the Derm7pt test dataset.
    else:
        test_mask_dir=None


In the output of this file, we have the MG4 metric related to the medical plausibility of the model, ranging from 0 to 1; in this case, the lower the better. In other words, it measures how well the explanation generated by the activation maps of the model's prototypes aligns with what is expected from a medical standpoint. The model should be more focused on the regions delimited as relevant in the medical context by the masks, rather than on those deemed non-relevant outside the medical context. For example, for PH2 and Derm7pt, 0 marks the interior of the skin lesion boundary and is therefore more relevant, while 1 marks what is outside the boundary and is therefore less relevant.

Furthermore, we obtain the X and Y data to create the balanced accuracy curve of the model based on the percentage of pixels that are removed and consequently set to black. This curve is important because, with the curve generated using the true heatmaps and the curve generated using random heatmaps, we can finally calculate the value of the MG3 metric, which takes into account the area under the curves. Refer to the file MG3_AREA_CURVE.ipynb to calculate, for a given obtained curve, the area under the curve when using the true heatmaps of the model (true curve) and the area under the curve when using randomly generated heatmaps (baseline curve), as well as the MG3 metric. In this case, the higher the MG3 value, the better. If the value is negative, it indicates that the explanation generated by the model is not superior to that generated by completely random maps. Please refer to Section 5.4, Guideline-Based Evaluation of Medical Image Analysis XAI, in the mentioned thesis.
