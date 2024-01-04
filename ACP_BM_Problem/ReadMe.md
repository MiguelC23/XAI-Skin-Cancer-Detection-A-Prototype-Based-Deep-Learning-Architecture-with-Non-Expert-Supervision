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


