# Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision
## Interpretable Skin Cancer Detection with Prototypes
Code Regarding the models associated with the section "Interpretable Skin Cancer Detection with Prototypes" of the Master's Thesis in Electrical and Computer Engineering titled 
"Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision," authored by Miguel Joaquim Nobre Correia conducted at the Instituto Superior Técnico, Lisbon, Portugal.

This code was used to train and evaluate the models identified in the results using the following loss functions $L_{\text{P}}$, $L_{\text{P}}+L_{\text{M}}$, $L_{\text{P}}+L_{\text{R}}$ and $L_{\text{P-Masked}}$.  

In this type of models, we have prototypes for all classes, not just prototypes of the malignant class. These models can be applied to both a binary problem, Melanoma vs Nevus, and a multiclass problem where we have 8 classes of skin lesions: AK, BCC, BKL, DF, MEL, NV, SCC, VASC.

The labels in the binary problem are 0 (MEL) and 1 (NV), while in the problem with the 8 classes of skin lesions, the labels are 0 (AK), 1 (BCC), 2 (BKL), 3 (DF), 4 (MEL), 5 (NV), 6 (SCC), and 7 (VASC).

Here is the meaning of each acronym for the classes:  
Melanoma (MEL)  
Melanocytic nevus (NV)  
Basal cell carcinoma (BCC)  
Actinic keratosis (AK) 
Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis also called lichenoid keratosis) (BKL)  
Dermatofibroma (DF)  
Vascular lesion (VASC)  
Squamous cell carcinoma (SCC)

## 1-How to train the model?

To train the model, simply run the main.py file. It is not necessary to provide arguments in the code line, as the model settings are edited in the settings.py file. After that, just run main.py.
Regarding the settings.py file, carefully examine the file as it is commented, explaining each of the parameters of the model settings. However, I would like to emphasize here that it is important to pay attention to the paths for the folders where the data is located.

Regarding the work carried out in the referenced thesis for training and validation, the images were sourced from the ISIC 2019 dataset. Whether it was the training or validation folders of the images or masks, they always contained the folders corresponding to the 8 classes AK, BCC, BKL, DF, MEL, NV, SCC, and VASC. Subsequently, depending on the user's choice in the settings.py file, with num_classes = 2 or num_classes = 8, the code will selectively retrieve only the folders for MEL and NV or all of them, respectively.

Furthermore, although it was not utilized in the thesis under consideration but rather in subsequent work post-publication. There exists the possibility of addressing the MEL vs. NV binary classification problem using the same validation set, but during training, the images of Melanoma (MEL) and Nevus (NV) from the ISIC 2019 dataset can be combined with those from the EDEASD (EASY Dermoscopy Expert Agreement Study Dataset). Let's identify this scenario as BP Train ISIC 2019 + EDEASD, where BP stands for Binary Problem.

(train_dir)-Path to the training set; in our case, it refers to the training images from ISIC 2019. Another option is to also include the training images from SIC 2019 along with the images from the EDEASD dataset, but in this latter case, it is only applied to the binary problem.

    train_dir ="..." 
    
(test_dir)-Path to the validation set; in our case, it refers to the validation images from ISIC 2019.

    test_dir ="..." 

(train_push_dir)-In our case, it remains the same as train_dir when train_dir is Train ISIC 2019. However, when train_dir is BP Train ISIC 2019 + EDEASD, we can restrict it to only EDEASD. This is because EDEASD contains dermatological concept annotations, allowing us to determine the true concept represented by the prototype.

    train_push_dir ="..." 

(train_mask_dir)-Path to binary masks of size 224x224x1, one for each training image, with the same name as the corresponding image. For example, the image ISIC_0000013.JPG has the mask ISIC_0000013.PNG. These masks identify pixels relevant from a medical perspective, i.e., within the skin lesion boundary (labeled 0) or outside the boundary (labeled 1), similar to skin lesion segmentation. 
When EDEASD images are used in training (BP Train ISIC 2019 + EDEASD), each mask identifies pixels associated with 1 or more medical concepts by at least 1 doctor or by a minimum of 3 doctors (labeled 0), depending on the chosen user's level of stringency and pixels that are not associated with any concept are marked with 0. 0 always identifies what is relevant or important in the image and 1 what is not relevant or important.

    train_mask_dir="..." 
    
(test_mask_dir)-Masks for the validation set images. It is not mandatory but necessary to use when LP_MASKED=True.

    test_mask_dir="..."

(forbidden_protos_directory)- Folder containing prohibited prototypes, meaning models should not learn these prototypes and therefore "remember" them. The information within this folder is only utilized in the following coefficients configuration, where 'LF' takes a nonzero value. This configuration corresponds to the $L_{\text{P}}+L_{\text{F}}$ where we use the forgeting loss that can be utilized

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
        'LM': 0,
        'LF': 0.09,# For example
        'LR': 0,
    };
        
however, in the case of the referenced thesis, we focused solely on the scenario involving only remembering loss ($L_{\text{P}}+L_{\text{R}}$). In this case, when the 'LR' coefficient takes a non-zero value, the information present in the 'remembering_protos_directory' is utilized. This directory corresponds to the path where we store the prototypes per class chosen by the user as examples that should be learned, or in other words, remembered by the model.

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
        'LM': 0,
        'LF': 0,
        'LR': 0.02,
    }

    forbidden_protos_directory=r"C:\ACP_BM_Problem\FP" 
    remembering_protos_directory=r"C:\ACP_BM_Problem\RP"

Within the folders of prohibited prototypes and those that must be remembered, you will find images of skin lesions with dimensions of 224x224x3. These images are organized into folders based on their class, such as class_idx_0 and class_idx_1 for binary classification problems, or class_idx_0, class_idx_1, class_idx_2, class_idx_3, class_idx_4, class_idx_5, class_idx_6, and class_idx_7 for the case of a problem with 8 classes. Within these subfolders, you'll find images in the following format, for example, ISIC_0010349_31.png, where ISIC_0010349 represents the identifying name of the image, and 31 represents the index of the patch that represents a prototype.

The prototypes present in these folders will be loaded by the code, and data will be used as input information to the model to apply forgetting loss and remembering loss, meaning they are only utilized when coefs['LF'] or coefs['LR'] in settings.py are nonzero. It is important to note that the prototypes in these folders can be obtained by running the debug.py file. After the user has executed models with the $L_{\text{P}}$ and $L_{\text{P}}+L_{\text{M}}$ scenarios, running the debug.py file on those models allows the user to obtain examples of prototypes that are considered prohibited and examples of prototypes that are considered relevant. These relevant prototypes are considered good examples for each class and should be remembered. In the case of the referenced thesis, we focused solely on prototypes that should be remembered, as we concentrated on the $L_{\text{P}}+L_{\text{R}}$ scenario. For the binary problem, we had 25 examples of prototypes chosen by the user per class in the remembering_protos_directory folder, and 6 per class in the multiclass problem.

## 2-Output of the code after running main.py and thus having trained the model.
After training, a folder named NC2(2 classes) or NC8(8 classes) will be created in the directory where the code is located. Within NC2 or NC8, you can find a folder named resnet18 (if that was the base CNN used). Inside the resnet18 folder, there is a folder with the name of the run. Within this folder, you can find the obtained prototypes and the checkpoints with the saved models, with a .pth extension.

## 3-How to obtain prototype examples for utilizing forgetting or remembering loss? Run the debug.py file.
After training some models related to scenarios $L_{\text{P}}$ and $L_{\text{P}}+L_{\text{M}}$, and having executed the global_analysis_train_dataset.py file for these models, we will proceed to run the debug.py file. 
To do so, there is no need to provide command line arguments; simply run the code after editing the following lines as necessary.

    if __name__ == '__main__':
    ...
    
    path_to_folder=r"C:\ACP_BM_Problem\NC2\resnet18\run1\20_1push0.8331_nearest_train" #Path to the folder created for the model after running the global_analysis_train_dataset.py file,
    #wherein the debug.py will be executed on the model to obtain examples of prototypes to prohibit and prototypes to remember.
    path_to_model_dir=r"C:\ACP_BM_Problem\NC2\resnet18\run1" path to the folder where is the model
    
    interactive= True #Put this True first, after run it again but with false
    move=not interactive

    num_classes=2 # Number total of classes, in our case 2 when MEL VS NEVUS or 8 when all classes from ISIC 2019
    ...
    #'-n-img', type=int, default=10, help='number of nearest patches to show for each prototype'
    n_img=10

On the first run, execute the file with `interactive=True` so that the user can inspect prototypes and designate each as prohibited by typing 'y' in the terminal, or as a reminder by typing 'r'. Subsequently, run the file again with `interactive=False` to relocate the prohibited prototypes and those to be remembered to the directories specified in settings.py:

    forbidden_protos_directory = r"C:\ACP_BM_Problem\FP"
    remembering_protos_directory = r"C:\ACP_BM_Problem\RP"

## 4- How to test the model on the PH2 or Derm7pt test sets?
Simply run the file TestPH2_or_Derm7pt.py, and there is no need to provide arguments in the code line; just edit the following lines.
    
    path_to_one_model=r"...pth" #path to the model you want to test
    VALMASKS=True # Set this to true if you have masks for the validation set. 
    MASKS_IN_TEST_SETS=True # Set this to true if you have masks for the validation set and if you trained the model with LP_MASKED=True.
    # meaning the model discards patches from images associated with areas marked as 1 by the masks, i.e., areas that are not directly important in 
    #the internal structure of the model during the forward process. Therefore, during testing or validation, masks for these images should also be provided to the model. 
    #In other words, the model makes decisions solely based on patches associated with relevant areas marked as 0 by the masks. 
    #If the model was trained with LP_MASKED=False, set this line to false as well.

    PH2=True# If true test in PH2 else test in Derm7pt

If PH2=True and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=True.

If PH2=False and the model was trained with 8 classes, set FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=True and FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False.

    #This flags are important to show only results relate to the classes present in the Dataset we are testing
    #Because PH2 only has 2 classes and Derm7pt 6 classes
    #But if we trained the model with 8 classes need to pay attention to this flags
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_DERM7PT=False # Always false if the model has been trained on the binary problem.
    FLAG_MODEL_TRAINED_WITH_8_CLASSES_AND_TEST_PH2=False #Always false if the model has been trained on the binary problem.

    ...
    VAL_dir=r"..." # Path to ISIC 2019 validation dataset
    if(VALMASKS==True):
        VAL_masks_path=r"..." # Path to ISIC 2019 validation masks dataset. O marks relevant pixels and 1 what is not relevant. They are only used when VALMASKS=True.
    else:
        VAL_masks_path=None
    
    ...
    if(PH2==True):
        test_dir=r"..." #Path to the 224x224x3 images of the PH2 test set.
        folder=r"...\ACP_BM_PROBLEM\PH2_Results" #Path to the folder where we will save a text file containing the results of testing the model on the PH2 test set.
    else:
        test_dir=r"..." #Path to the 224x224x3 images of the Derm7pt test set.
        folder=r"...\ACP_BM_PROBLEM\Derm7pt_Results" #Path to the folder where we will save a text file containing the results of testing the model on the Derm7pt test set.
    ...
    if(PH2==True):
        test_mask_dir=r"..." Path to PH2 masks. O marks relevant pixels and 1 what is not relevant. They are only used when MASKS_IN_TEST_SETS=True.
    elif(PH2==False):
        test_mask_dir=r"..." Path to Derm7pt masks. O marks relevant pixels and 1 what is not relevant. They are only used when MASKS_IN_TEST_SETS=True.

The image folder of the PH2 dataset has two subfolders, one for each class (MEL and NV), and the same applies to the folder containing the masks. In the case of the Derm7pt dataset, the image folder contains 6 subfolders, each corresponding to a class (BCC, BKL, DF, MEL, NV, VASC), and the same structure is present in the masks folder. It is worth noting that the images have dimensions of 224x224x3, while the masks have dimensions of 224x224x1. The masks serve to identify important/relevant pixels (inside the boundary of the skin lesion) in this medical context, represented by 0, and non-relevant/unimportant pixels (outside the boundary of the skin lesion) are represented by 1.

## 5-See the importance of each prototype for the model.
Simply run the file "Assessment_of_the_weight_of_each_prototype.py"; no need to provide arguments in the terminal command line. The weight of each prototype is determined by the difference in balanced accuracy on the ISIC 2019 validation set when each prototype is removed individually.
Change only the following lines of code according to what is necessary.

    load_model_path=r"....pth" Path to the model you want to evaluate.
    folder_path_to_save_results=r"...\ACP_BM_Problem\Weights_of_P" #Path to the folder where a .txt file with the results will be saved.
    
    MASKS_IN_TEST_SETS=True #We recommend setting it to True if the model has been trained with LP_MASKED=True, meaning the model discards patches 
    #from images associated with areas marked as 1 by the masks, i.e., areas that are not directly important in the internal structure of the model during the forward process. 
    #Therefore, during testing or validation, masks for these images should also be provided to the model. In other words, the model makes decisions solely based on patches
    #associated with relevant areas marked as 0 by the masks. If the model was trained with LP_MASKED=False, set this line to false as well
    if(MASKS_IN_TEST_SETS==True):
        test_mask_dir=r"..."#Path to the folder where the masks for the validation set are located. Only used when MASKS_IN_TEST_SETS=True.
    else:
        test_mask_dir=None

## 6-How to observe the explanation generated by the model in the Binary Probelm MEL vs NV for an image?
The first step is to run the local_analysis_skin.py file, and then you should run the local_analysis_vis.py file.
You don't need to provide arguments in the code line in the terminal. Instead, edit the following lines of code in the file local_analysis_skin.py

    most_activated_prototypes=... #chosse the same number as prototypes if you want.

    Test_Dataset_variable=0 # Choose the Dataset from you 0 ISIC 2019; 1 PH2; 2 DERM7PT 
    # Choose the dataset from which you want the image to be selected to visualize the model-generated explanation 
    #for classifying that image as either melanoma or nevus. Each dataset is associated with an integer number 0, 1, or 2.

    MASKS_IN_TEST=... "Set this to true if the model was trained with LP_MASKED=True."
    ...
    if(inp==0):
        if(Test_Dataset_variable==0):
            test_image_dir = r"...Path to the folder containing the images of the melanoma class in the ISIC 2019 validation set..."
            mask_dir=r"......Path to the folder containing the masks of the melanoma class in the ISIC 2019 validation set..."
        elif(Test_Dataset_variable==1):
            test_image_dir = r"...Path to the folder containing the images of the melanoma class in the PH2 test set..."
            mask_dir=r"...Path to the folder containing the masks of the melanoma class in the PH2 test set..."
        elif(Test_Dataset_variable==2):
            test_image_dir = r"...Path to the folder containing the images of the melanoma class in the Derm7pt test set..."
            mask_dir=r"...Path to the folder containing the masks of the melanoma class in the Derm7pt test set..."
    ...        

    ...
    if(inp==1):
        if(Test_Dataset_variable==0):
            test_image_dir = r"...Path to the folder containing the images of the nevus class in the ISIC 2019 validation set..."
            mask_dir=r"......Path to the folder containing the masks of the nevus class in the ISIC 2019 validation set..."
        elif(Test_Dataset_variable==1):
            test_image_dir = r"...Path to the folder containing the images of the nevus class in the PH2 test set..."
            mask_dir=r"...Path to the folder containing the masks of the nevus class in the PH2 test set..."
        elif(Test_Dataset_variable==2):
            test_image_dir = r"...Path to the folder containing the images of the nevus class in the Derm7pt test set..."
            mask_dir=r"...Path to the folder containing the masks of the nevus class in the Derm7pt test set..."
    ...
    
    ...
    load_model_dir = r"...Path to folder where the model is" 
    load_model_name = r"...name of the model..."

Edit the following lines of file local_analysis_vis.py as well:

    source_dir = r"...Path to the folder created for the local analysis of an image after running local_analysis_skin.py..." 
    number_prototypes_in_figure=... # Number indicating the number of prototypes appearing in the explanatory image; it can be the number of model prototypes
    
## 7-How to observe the explanation generated by the model in the Multiclass Probelm (8 classes 0: 'AK', 1:'BCC', 2:'BKL', 3:'DF', 4:'MEL', 5:'NV', 6:'SCC', 7:'VASC') for an image?
The first step is to run the local_analysis_skin8C.py file, and then you should run the local_analysis_vis_8C.py file.
You don't need to provide arguments in the code line in the terminal. Instead, edit the following lines of code in the file local_analysis_skin8C.py

    most_activated_prototypes=... #chosse the same number as prototypes if you want.
    # specify the test image to be analyzed
    Test_Dataset_variable=0 # 0 ISIC 2019; 1 PH2; 2 DERM7PT
    # Choose the dataset from which you want the image to be selected to visualize the model-generated explanation 
    #for classifying that image as 0: 'AK', 1:'BCC', 2:'BKL', 3:'DF', 4:'MEL', 5:'NV', 6:'SCC', 7:'VASC'. Each dataset is associated with an integer number 0, 1, or 2.
    MASKS_IN_TEST=... "Set this to true if the model was trained with LP_MASKED=True."
    ...
    if(Test_Dataset_variable==0):
        test_image_dir = r"Path\to\ISIC2019\VALIDATION\IMAGES\FOLDER..\\"+output[inp] #args.test_img_dir[0]
        mask_dir=r"Path\to\ISIC2019\VALIDATION\Masks\FOLDER..\\"+output[inp]
    elif(Test_Dataset_variable==1):
        test_image_dir = r"...Path to PH2 test images folder...\\"+output[inp]
        mask_dir=r"...Path to PH2 test masks folder..."+output[inp]
    elif(Test_Dataset_variable==2):
        test_image_dir = r"...Path to Derm7pt test images folder...\\"+output[inp]
        mask_dir=r"...Path to Derm7pt test Masks folder...\\"+output[inp] 
    ...
    ...
    load_model_dir = r"...Path to folder where the model is" 
    load_model_name = r"...name of the model..."

Edit the following lines of file local_analysis_vis_8C.py as well:

    source_dir = r"...Path to the folder created for the local analysis of an image after running local_analysis_skin.py..." 
    number_prototypes_in_figure=... # Number indicating the number of prototypes appearing in the explanatory image; it can be the number of model prototypes.

## 8-View the k patches from the training set closest to each prototype--->global_analysis_train_dataset.py
Simply run the global_analysis_train_dataset.py file; there is no need to provide arguments in the terminal, just edit the following lines of code:

    if __name__ == '__main__':
        load_model_path=r"..."# Path to the model file .pth
        k=... # Number of patches we want to see that are closest to each prototype. We may want to view the top 5, 10, or 20, for example.
        ...

## 9-View the k patches from the EDEASD (EASY Dermoscopy Expert Agreement Study dataset) set closest to each prototype. Only for MEL vs NV Problem! 
Simply run the **global_analysis_EDEASD.py** file; there is no need to provide arguments in the terminal, just edit the following lines of code:

    def main(load_model_path: str, k: int):
        ...
        train_push_dir=r"..." #Folder with only the EDEASD images, inside this folder we have two subfolders, one for each class, MEL and NV.

    if __name__ == '__main__':
        load_model_path=r"..."# Path to the model file .pth
        k=... # Number of patches we want to see that are closest to each prototype. We may want to view the top 5, 10, or 20, for example.
        ...

## 10- Examine, for each prototype, which patches of a given concept are closest to each prototype, as well as, for the patches of a concept that are closest to PX, the average distance of these patches to that prototype. 
**It only makes sense to be used when the MEL and NV images from EDEASD were used during training**. This is because only for these images do we have information at the level of annotations made by dermatologists regarding dermatological concepts pixel by pixel for each image. And when the push directory during training was exclusively the images from EDEASD. For this case, run the file **concept_evaluation_EDEASD.py**.
You don't need to provide arguments in the terminal command line; just edit the following lines before running the file:

    if __name__ == '__main__':
        ...
        ...
        train_push_dir = r"..." #Folder with only the EDEASD images, inside this folder we have two subfolders, one for each class, MEL and NV.
        load_model_path=r"....pth" #Path to the model you want to evaluate.
        size=...# Dimension of the maps that are the input of the prototype layer, i.e the output of convolution layers.
        #Example [BATCH_SIZE,D,P,P]. So you should put size=(P,P). Only VGG16 has (14,14) the others is (7,7)

    def concept_eval(model, dataloader, size=(7,7)):
        ...
        MIC3_masksfolder_path=r"..." #Path to the masks of EDEASD images of type MIC-3, meaning that for each image, there is a mask for each concept identified by dermatologists. 
        #In each mask, pixels representing the concept are identified with a minimum agreement of at least 3 physicians. In this case, the masks are not inverted, meaning that 
        #a pixel annotated with 1 represents something important or relevant, and 0 represents a pixel that has not been annotated and is not important or relevant.
        ...

## 11- Examine the concepts present in a given patch or prototype of an EDEASD image.
Simply run the file **true_concept_EDEASD_proto.py**; there is no need to provide arguments in the terminal command line. However, you should edit the following lines of code as needed. 

    if __name__ == '__main__':
        ...
        PROTO_ISIC_ID_EDEASD=... # Example 'ISIC_0046495'
        PATCH_IDX=25 # From 0 to 48 if the image is represented in the latent space by a 7x7 dimension map; otherwise, patches are numbered from 0 to 195 when the map has a dimension of 14x14. 
        ...
        map_size = ...# Example 7 or 14. 
        ....
    
    def concepts_present(ID,a,b,size):
        ...
        MIC3_masksfolder_path=r"..." #Path to the masks of EDEASD images of type MIC-3, meaning that for each image, there is a mask for each concept identified by dermatologists. 
        In each mask, pixels representing the concept are identified with a minimum agreement of at least 3 physicians. 
        In this case, the masks are not inverted, meaning that a pixel annotated with 1 represents something important or relevant, 
        and 0 represents a pixel that has not been annotated and is not important or relevant.
        ...

The code outputs the concepts present in the patch and an associated percentage indicating how much of the patch the concept occupies. The larger the area it covers, the higher the percentage.

## 12- Visualize the latent space representation of the model by creating a plot using t-SNE.
Simply run the file **Visualize_Latent_Space.py**; there is no need to provide arguments in the command line—just execute the file and edit the following lines of code:

    if __name__ == '__main__':
        images=False # If true each point is an image else each point is a patch
        filter_patches=True #  If True We only want relevant patchs marked with 0s in inverted masks. Only put True when images=False
        only_max_pool_patches=True # Select the most activated patch between an image and each prototype when True, as utilizing all patches in t-SNE from the training dataset proves to be computationally expensive.
        load_model_path=r"..." Path to the "//.pth" model
        size=(7,7) # Dimension of the maps that are the input of the prototype layer, i.e the output of convolution layers. Example [BATCH_SIZE,D,P,P]. So you should put size=(P,P). Only VGG16 has (14,14) the others is (7,7)
