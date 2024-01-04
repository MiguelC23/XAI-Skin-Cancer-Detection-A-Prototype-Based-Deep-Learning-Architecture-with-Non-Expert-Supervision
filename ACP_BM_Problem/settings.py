base_architecture = 'resnet18' # Choose the first main component of the model a CNN:'resnet18','resnet50','densenet169','vgg16','eb3'
img_size = 224 # The input images for the model have dimensions of 224x224x3.
latent_shape = 256#Latent space dimension, choose between 128, 256, or 512.
number_of_prototypes_per_class=2#Choose the number of prototypes for each class.
num_classes = 2# Choose 2 for MEL VS NEVUS. Or choose 8 to consider the following classes: AK BCC BKL DF MEL NV SCC VASC
prototype_shape = (int(num_classes*number_of_prototypes_per_class),latent_shape, 1, 1)#DO NOT ALTER THIS LINE
diversity=True #All prototypes are push to diferent patches there are not repeated prototypes if True. WE RECOMMEND SETTING IT TO TRUE.
topk_k=1 #Choose a number between 1 and 49 when the output of the CNN, that is, the input to the prototypes layer, is [BATCH_SIZE, D, 7, 7], or a value between 1 and 196 when the dimension is [BATCH_SIZE, D, 14, 14].

#The online augmentation used is transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) HorizontalFlip() VerticalFlip().
online_augmentation=True #We recommend leaving "True" since it significantly aids in the model's generalization when applied to test sets, such as PH2 and Derm7pt in our case.
#Inverted masks are designed such that a value of 1 indicates what is not relevant or important, while a value of 0 signifies what is important or relevant.

LP_MASKED=True #Put True to Use masks in the internal structure of the model, i.e in the forward process to disregard patches that are not relevant.
Fixed_prototypes_during_training_initialized_orthogonally=True # True if you want to keep the prototypes fixed during training and orthogonal,
#which means that only the latent space of images/patches is updated around the prototypes. 
#In this case, the push does not perform an assignment because it does not replace the latent representation 
#of the prototype with the training patch closest to that representation. 
#It only allows visualizing the training patch closest to the latent representation of the prototype.

class_specific_penalization=False
last_layer_weight=-1 # Initialization of the last layer(classifier) weights.
random_seed_number=4 #Random seed choose by the use for Reproducibility purposes


prototype_activation_function = "log"
prototype_activation_function_in_numpy = prototype_activation_function
add_on_layers_type = 'regular'

folder_path_to_save_runs=r"C:\ACP_BM_Problem\NC"+str(num_classes)

train_dir =r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train"
test_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val"
train_push_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train"
train_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train_Fine_masks"
test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks" #only affects when LP_MASKED=True

#train_dir =r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\big_bea_plus_doctors\train"#"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train"
#test_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val"
#train_push_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224_sum_all_concept_agree3\images_for_this_masks\MEL_NV"#"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train"
#train_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\big_bea_plus_doctors\fine_masks_evenjust1doc"#"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train_Fine_masks"
#test_mask_dir=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val_Fine_masks" #only affects when LP_MASKED=True

#DO NOT ALTER THIS 4 LINES
if(test_mask_dir is not None):
    Test_results_with_test_masks=True
else:
    Test_results_with_test_masks=False


forbidden_protos_directory=r"C:\ACP_BM_Problem\FP" 
remembering_protos_directory=r"C:\ACP_BM_Problem\RP"

#path if you want to start from a model already learned and continue training if you dont want start from an already trained model put None
load_model_path=None

train_batch_size = 75
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 2e-4,#2e-4
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3



coefs = {
    'crs_ent': 1,
    'clst': 0.8,#0.8 
    'sep': -0.08,#-0.08
    'l1': 1e-4,
    'LM': 0,#0.001 
    'LF':0,#0.09
    'LR':0,#0.02
}

num_train_epochs = 21#21
num_warm_epochs = 5#5
push_start = 5#5
push_epochs=[5,10,15,20]#5,10,15,20
number_iterations=10 #10


experiment_run="run1"

