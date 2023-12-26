import os
import matplotlib
import numpy as np
matplotlib.use("Agg")
import torch
import torch.utils.data
import random
from settings import random_seed_number

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(random_seed_number)
torch.cuda.manual_seed(random_seed_number)
np.random.seed(random_seed_number)
random.seed(random_seed_number)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


def LICD(prototypes, num_classes):
    prototypes_per_class = prototypes.shape[0] // num_classes #Only malignant prototypes are used. For example for 18 prototypes we have 9 of MEL and 9 of NV, but only MEL matters, because only MEL is used for classification.
    class_prototypes = []
    for i in range(num_classes):
        start_idx = i * prototypes_per_class
        end_idx = (i + 1) * prototypes_per_class
        class_prototypes.append(prototypes[start_idx:end_idx])
    
    A = class_prototypes[0] #[9,256,1,1]
    matriz_distancias=torch.zeros(prototypes_per_class, prototypes_per_class)
    for a in range(prototypes_per_class):
        for b in range(prototypes_per_class):
            # Resize the vectors to [N, D] where N is the number of vectors and D is the dimension.
            PA=A[a].squeeze() #[256]
            PB=A[b].squeeze() #[256]
            PA = PA.unsqueeze(0) #[1,256]
            PB = PB.unsqueeze(0) #[1,256]
            #Calculate the Euclidean distance between vectors using torch.cdist
            distancia_euclidiana = torch.cdist(PA,PB)
            matriz_distancias[a,b]=distancia_euclidiana
    
    max_distancia = torch.max(matriz_distancias)
    # Normalize the matrix by dividing each element by the maximum value
    matriz_distancias  = matriz_distancias / max_distancia
    # distance_matrix is a symmetric matrix
    # Summing the values in the upper triangular part (excluding the diagonal)
    soma_distancias = torch.sum(torch.triu(matriz_distancias, diagonal=1))
    numero_distancias=((prototypes_per_class*prototypes_per_class)-prototypes_per_class)/2
    media=soma_distancias/numero_distancias
    return media

def VICN(prototypes, num_classes):
    prototypes_per_class = prototypes.shape[0] // num_classes
    class_prototypes = []
    for i in range(num_classes):
        start_idx = i * prototypes_per_class
        end_idx = (i + 1) * prototypes_per_class
        class_prototypes.append(prototypes[start_idx:end_idx])

    A = class_prototypes[0]  # Assuming the first class prototypes are of interest

    # Convert the PyTorch tensor to a NumPy array for calculations
    points_array = A.view(prototypes_per_class, -1).detach().cpu().numpy()

    # Calculate the centroid of the cluster
    centroid = np.mean(points_array, axis=0)

    # Calculate the squared distances of the points from the centroid
    distances = np.linalg.norm(points_array - centroid, axis=1) ** 2
    # Normalize the squared distances by the maximum value
    normalized_distances = distances / np.max(distances)

    # Calculate the average normalized intra-cluster variance
    intra_cluster_variance = np.mean(normalized_distances)

    return intra_cluster_variance


if __name__ == '__main__':


    load_model_path=r"C:\PATH\TO\YOUR\MODDEL.pth"
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    num_classes=ppnet_multi.module.num_classes
    num_prototypes=ppnet_multi.module.num_prototypes
    licd_value=LICD(ppnet_multi.module.prototype_vectors,ppnet_multi.module.num_classes)
    vicn_value=VICN(ppnet_multi.module.prototype_vectors,ppnet_multi.module.num_classes)

    
    print("intra-class diversity loss (LICD):",licd_value)
    print("Normalized intra-class variance (VICN):",vicn_value)


