# Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision
## One-Class Prototypes: Simplified Binary Problem Explanation

Code Regarding the models associated with the section "One-Class Prototypes: Simplified Binary Problem Explanation" of the Master's Thesis in Electrical and Computer Engineering titled "Explainable Artificial Intelligence for Skin Cancer Detection: A Prototype-Based Deep Learning Architecture with Non-Expert Supervision," authored by Miguel Joaquim Nobre Correia conducted at the Instituto Superior Técnico, Lisbon, Portugal.

This code was used to train and evaluate the models identified in the results using the following loss functions LP-1C, LP-1C-Masked, and LP-1C-Masked+LICD. In this type of models, the decision is made solely based on the prototypes of the malignant class, namely melanoma (MEL), with the prototypes of the benign class, namely nevus (NV), not used in the classification process. In other words, only the similarities with the melanoma prototypes are provided as input to the final fully connected layer of the model, the classifier.

This code was designed solely for application to a binary classification problem of skin cancer lesions with two classes. It has been exclusively used to distinguish between melanoma and nevus (MEL vs NV), with MEL identified as class 0 and NV as class 1.

## 1-How to train the model?

To train the model, simply run the main.py file. It is not necessary to provide arguments in the code line, as the model settings are edited in the settings.py file. After that, just run main.py.
Regarding the settings.py file, carefully examine the file as it is commented, explaining each of the parameters of the model settings. However, I would like to emphasize here that it is important to pay attention to the paths for the folders where the data is located.

(train_dir)-Path to the training set; in our case, it refers to the training images from ISIC 2019. Another option is to also include the training images from SIC 2019 along with the images from the EDEASD dataset. In other words, the 1st option is Train ISIC 2019, and the 2nd option is Train ISIC 2019 + EDEASD. This second option was not applied in the referenced thesis, but rather in future work developed subsequently after the thesis was published.

    train_dir ="..." 
    
(test_dir)-Path to the validation set; in our case, it refers to the validation images from ISIC 2019.

    test_dir ="..." 

(train_push_dir)-In our case, it remains the same as train_dir when train_dir is Train ISIC 2019. However, when train_dir is Train ISIC 2019 + EDEASD, we can restrict it to only EDEASD. This is because EDEASD contains dermatological concept annotations, allowing us to determine the true concept represented by the prototype.

    train_push_dir ="..." 

(train_mask_dir)-Path to binary masks of size 224x224x1, one for each training image, with the same name as the corresponding image. For example, the image ISIC_0000013.JPG has the mask ISIC_0000013.PNG. These masks identify pixels relevant from a medical perspective, i.e., within the skin lesion boundary (labeled 0) or outside the boundary (labeled 1), similar to skin lesion segmentation. 
When EDEASD images are used in training (2nd option), each mask identifies pixels associated with 1 or more medical concepts by at least 1 doctor or by a minimum of 3 doctors (labeled 0), depending on the chosen user's level of stringency and pixels that are not associated with any concept are marked with 0. 0 always identifies what is relevant or important in the image and 1 what is not relevant or important.

    train_mask_dir="..." 
    
(test_mask_dir)-Masks for the validation set images. It is not mandatory but necessary to use when LP1C_MASKED=True.

    test_mask_dir="..."

All folders indicated in these paths always contain two folders, one for the Melanoma class named MEL, and one for the Nevus class named NV. Whether it is a path for images or a path for masks.

## 2-Output of the code after running main.py and thus having trained the model.
After training, a folder named NC2 will be created in the directory where the code is located. Within NC2, you can find a folder named resnet18 (if that was the base CNN used). Inside the resnet18 folder, there is a folder with the name of the run. Within this folder, you can find the obtained prototypes and the checkpoints with the saved models, with a .pth extension.

## 3- How to test the model on the PH2 or Derm7pt test sets?
Simply run the file TestPH2_or_Derm7pt.py, and there is no need to provide arguments in the code line; just edit the following lines.
    
(path_to_one_model)-path to the model you want to test
     
    path_to_one_model=r"....pth"
    
(MASKS_IN_TEST_SETS)-We recommend setting it to True if the model has been trained with LP1C_MASKED=True, meaning the model discards patches from images associated with areas marked as 1 by the masks, i.e., areas that are not directly important in the internal structure of the model during the forward process. Therefore, during testing or validation, masks for these images should also be provided to the model. In other words, the model makes decisions solely based on patches associated with relevant areas marked as 0 by the masks. If the model was trained with LP1C_MASKED=False, set this line to false as well.

    MASKS_IN_TEST_SETS=True 
    
(PH2)-If true test in PH2 else test in Derm7pt

    PH2=True

    if(PH2==True):
        test_dir=r"..." #Path to the 224x224x3 images of the PH2 test set.
        folder=r"...\1CP_BinaryProblem\PH2_Results" #Path to the folder where we will save a text file containing the results of testing the model on the PH2 test set.
    else:
        test_dir=r"..." #Path to the 224x224x3 images of the Derm7pt test set.
        folder=r"...\1CP_BinaryProblem\Derm7pt_Results" #Path to the folder where we will save a text file containing the results of testing the model on the Derm7pt test set.
    
    ...
    ...
    ...
    
    
    if(PH2==True):
        test_mask_dir=r"..." Path to PH2 masks. O marks relevant pixels and 1 what is not relevant. They are only used when MASKS_IN_TEST_SETS=True.
    elif(PH2==False):
        test_mask_dir=r"..." Path to Derm7pt masks. O marks relevant pixels and 1 what is not relevant. They are only used when MASKS_IN_TEST_SETS=True.

## 4-See the importance of each prototype for the model.
Simply run the file "Assessment_of_the_weight_of_each_prototype.py"; no need to provide arguments in the terminal command line. The weight of each prototype is determined by the difference in balanced accuracy on the ISIC 2019 validation set when each prototype is removed individually.
Change only the following lines of code according to what is necessary.

    load_model_path=r"....pth" Path to the model you want to evaluate.

    folder_path_to_save_results=r"...\1CP_BinaryProblem\Weights_of_P" #Path to the folder where a .txt file with the results will be saved.

(MASKS_IN_TEST_SETS)-We recommend setting it to True if the model has been trained with LP1C_MASKED=True, meaning the model discards patches from images associated with areas marked as 1 by the masks, i.e., areas that are not directly important in the internal structure of the model during the forward process. Therefore, during testing or validation, masks for these images should also be provided to the model. In other words, the model makes decisions solely based on patches associated with relevant areas marked as 0 by the masks. If the model was trained with LP1C_MASKED=False, set this line to false as well

    MASKS_IN_TEST_SETS=True 

    if(MASKS_IN_TEST_SETS==True):
        test_mask_dir=r"..." #Path to the folder where the masks for the validation set are located. Only used when MASKS_IN_TEST_SETS=True.
    else:
        test_mask_dir=None

## 5-How to observe the explanation generated by the model for an image?
The first step is to run the local_analysis_skin.py file, and then you should run the local_analysis_vis.py file.
You don't need to provide arguments in the code line in the terminal. Instead, edit the following lines of code in the file local_analysis_skin.py

    most_activated_prototypes=... #chosse the same number as prototypes if you want.

    Test_Dataset_variable=0 # Choose the Dataset from you 0 ISIC 2019; 1 PH2; 2 DERM7PT 
    # Choose the dataset from which you want the image to be selected to visualize the model-generated explanation 
    #for classifying that image as either melanoma or nevus. Each dataset is associated with an integer number 0, 1, or 2.

    MASKS_IN_TEST=... "Set this to true if the model was trained with LP1C_MASKED=True."
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
    number_prototypes_in_figure=... # Number indicating the number of prototypes appearing in the explanatory image; it can be the number of model prototypes.

## 6-View the k patches from the training set closest to each prototype. 
Simply run the global_analysis_train_dataset.py file; there is no need to provide arguments in the terminal, just edit the following lines of code:

    if __name__ == '__main__':
        load_model_path=r"..."# Path to the model file .pth
        k=... # Number of patches we want to see that are closest to each prototype. We may want to view the top 5, 10, or 20, for example.
        ...

## 7-View the k patches from the EDEASD (EASY Dermoscopy Expert Agreement Study dataset) set closest to each prototype. 
Simply run the global_analysis_EDEASD.py file; there is no need to provide arguments in the terminal, just edit the following lines of code:

    train_push_dir=r"..." #Folder with only the EDEASD images, inside this folder we have two subfolders, one for each class, MEL and NV.

    if __name__ == '__main__':
        load_model_path=r"..."# Path to the model file .pth
        k=... # Number of patches we want to see that are closest to each prototype. We may want to view the top 5, 10, or 20, for example.
        ...

## 8- Examine, for each prototype, which patches of a given concept are closest to each prototype, as well as, for the patches of a concept that are closest to PX, the average distance of these patches to that prototype.
It only makes sense to be used when the MEL and NV images from EDEASD were used during training. This is because only for these images do we have information at the level of annotations made by dermatologists regarding dermatological concepts pixel by pixel for each image. And when the push directory during training was exclusively the images from EDEASD. For this case, run the file concept_evaluation_EDEASD.py.
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

## 9- Examine the concepts present in a given patch or prototype of an EDEASD image.
Simply run the file true_concept_EDEASD_proto.py; there is no need to provide arguments in the terminal command line. However, you should edit the following lines of code as needed. 

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

## 10- Visualize the latent space representation of the model by creating a plot using t-SNE.
Simply run the file Visualize_Latent_Space.py; there is no need to provide arguments in the command line—just execute the file and edit the following lines of code:

    if __name__ == '__main__':
        images=False # If true each point is an image else each point is a patch
        filter_patches=True #  If True We only want relevant patchs marked with 0s in inverted masks. Only put True when images=False
        only_max_pool_patches=True # Select the most activated patch between an image and each melanoma prototype when True, as utilizing all patches in t-SNE from the training dataset proves to be computationally expensive.
        load_model_path=r"..." Path to the "//.pth" model
        size=(7,7) # Dimension of the maps that are the input of the prototype layer, i.e the output of convolution layers. Example [BATCH_SIZE,D,P,P]. So you should put size=(P,P). Only VGG16 has (14,14) the others is (7,7)

## 11- Conduct the evaluation of the following XAI metrics: G3-Truthfulness and G4-Informative Plausibility.
Just run the following file XAI_MG3_MG4.py; no need to provide arguments in the terminal command line, but you should edit the following lines as necessary:

    path_to_one_model=r"..." # Path to the model.
    MASKS_IN_TEST_SETS=... #Set this to true if the model was trained with LP1C_MASKED=True.
    PH2=True# If true test in PH2 else test in Derm7pt
    RANDOM_HEAT_MAPS=False# When true, perform the evaluation based on random heatmaps rather than the true model heatmaps, used for comparative purposes.
    ...
    if(PH2==True):
        test_dir=r"...Path to the folder containing the MEL and NV folders of the test images from the PH2 dataset..."
        folder=r"...\1CP_BinaryProblem\PH2_Results" #Path to the folder where a text file with the results will be saved.
    else:
        test_dir=r"...Path to the folder containing the MEL and NV folders of the test images from the Derm7pt dataset..."
        folder=r"...\1CP_BinaryProblem\Derm7pt_Results"#Path to the folder where a text file with the results will be saved.
    ...
    ...
    if(PH2==True):
        test_mask_dir=r"..." #Path to the folder where two folders are located, MEL and NV, i.e., the binary masks where 1 represents the non-relevant and 0 represents the relevant. 
        #One binary mask per image. 
        #In other words, the masks are inverted. For the PH2 test dataset.
    elif(PH2==False):
        test_mask_dir=r"..."  Path to the folder where two folders are located, MEL and NV, i.e., the binary masks where 1 represents the non-relevant and 0 represents the relevant. 
        #One binary mask per image. 
        #In other words, the masks are inverted. For the Derm7pt test dataset.
    ...

In the output of this file, we have the MG4 metric related to the medical plausibility of the model, ranging from 0 to 1; in this case, the lower the better. In other words, it measures how well the explanation generated by the activation maps of the model's prototypes aligns with what is expected from a medical standpoint. The model should be more focused on the regions delimited as relevant in the medical context by the masks, rather than on those deemed non-relevant outside the medical context. For example, for PH2 and Derm7pt, 0 marks the interior of the skin lesion boundary and is therefore more relevant, while 1 marks what is outside the boundary and is therefore less relevant.

Furthermore, we obtain the X and Y data to create the balanced accuracy curve of the model based on the percentage of pixels that are removed and consequently set to black. This curve is important because, with the curve generated using the true heatmaps and the curve generated using random heatmaps, we can finally calculate the value of the MG3 metric, which takes into account the area under the curves. Refer to the file MG3_AREA_CURVE.ipynb to calculate, for a given obtained curve, the area under the curve when using the true heatmaps of the model (true curve) and the area under the curve when using randomly generated heatmaps (baseline curve), as well as the MG3 metric. In this case, the higher the MG3 value, the better. If the value is negative, it indicates that the explanation generated by the model is not superior to that generated by completely random maps. Please refer to Section 5.4, Guideline-Based Evaluation of Medical Image Analysis XAI, in the mentioned thesis.

## 12-Perform an analysis of the concepts that a prototype may be triggering/activating using the information from the top-k patches of EDEASD closest to that prototype. 
Run the file "Patches_EDEASD_closer_to_PX_concept_analysis.py" after executing the "global_analysis_EDEASD.py" file.
Edit the following lines of code in the file Patches_EDEASD_closer_to_PX_concept_analysis.py:

    ...
    pasta = r"...\1CP_BinaryProblem\NC2\resnet18\run1\20_0push0.7869_nearest_from_EDEASD\0" # Example of the path to the folder of the prototype 0 created after runing the file global_analysis_EDEASD.py
    ...
    caminho_do_arquivo_csv = r"...metadata_concepts.csv" Csv with concept metadata of the EDEASD images.
    ...
    caminho_masks=r"..." Path to the folder containing the masks created by the doctors for the EDEASD images. 
    #The masks designate a pixel with a value of 1 for the corresponding concept identified by the doctor in that mask, and a pixel with a value of 0 for the areas that do not represent the identified concept in that mask. 
    #Within the folder, you will find masks for all the images. It is important to understand that each mask corresponds to an image, a doctor, and one concept. 
    #Consequently, there are multiple masks for the same image because different concepts may be present in the same image. Additionally, we may have different 
    #masks for the same image and concept created by different doctors. The masks have dimension 224x224x1.

