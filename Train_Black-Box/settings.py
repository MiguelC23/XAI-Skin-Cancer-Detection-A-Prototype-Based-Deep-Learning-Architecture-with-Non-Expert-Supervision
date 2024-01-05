import torch
import numpy as np

base_architecture = 'resnet18' # Choose the CNN:'resnet18','resnet50','densenet169','vgg16','eb3'
img_size = 224 # The input images for the model have dimensions of 224x224x3.
num_classes = 2 # Choose 2 for MEL VS NEVUS. Or choose 8 to consider the following classes: AK BCC BKL DF MEL NV SCC VASC
random_seed_number=4
#The online augmentation used is transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) HorizontalFlip() VerticalFlip().
OA=True #We recommend leaving "True" since it significantly aids in the model's generalization when applied to test sets, such as PH2 and Derm7pt in our case.

train_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\train"
test_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\Bea_LIMPO\limpo\val"

folder_path_to_save_runs=r"E:\Coisas da Tese\Train_Black-Box\\NC"+str(num_classes)

load_model=False
load_model_path=None

train_batch_size = 250
test_batch_size = 250

num_train_epochs = 100
lr=0.001
id=1

experiment_run=base_architecture+'_id_'+str(id)
