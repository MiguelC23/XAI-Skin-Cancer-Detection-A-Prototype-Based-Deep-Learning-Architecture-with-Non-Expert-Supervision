conceitos_numeros = {
    'Vessels_-_Comma': 0,
    'Vessels_-_Polymorphous': 1,
    'Network_-_Delicate_Pigment_Network_+_Reticulation': 2,
    'Network_-_Typical_pigment_network_+_Reticulation': 3,
    'Globules_+_Clods_-_Rim_of_brown_globules': 4,
    'Regression_structures_-_Peppering_+_Granularity': 5,
    'Regression_structures_-_Scarlike_depigmentation': 6,
    'Globules_+_Clods_-_Milky_red': 7,
    'Lines_-_Angulated_lines_+_Polygons_+_Zig-zag_pattern': 8,
    'Structureless_-_Blotch_regular': 9,
    'Lines_-_Branched_streaks': 10,
    'Structureless_-_Blue-whitish_veil': 11,
    'Structureless_-_Structureless_brown_tan': 12,
    'Dots_-_Irregular': 13,
    'Network_-_Atypical_pigment_network_+_Reticulation': 14,
    'Globules_+_Clods_-_Regular': 15,
    'Vessels_-_Linear_irregular': 16,
    'Globules_+_Clods_-_Irregular': 17,
    'Pattern_-_Starburst': 18,
    'Lines_-_Pseudopods': 19,
    'Dots_-_Regular': 20,
    'Lines_-_Radial_streaming': 21,
    'Pattern_-_Homogeneous_-_NOS': 22,
    'Shiny_white_structures_-_Shiny_white_streaks': 23,
    'Globules_+_Clods_-_Cobblestone_pattern': 24,
    'Structureless_-_Milky_red_areas': 25,
    'Structureless_-_Blotch_irregular': 26,
    'Network_-_Negative_pigment_network': 27,
    'Network_-_Broadened_pigment_network_+_Reticulation': 28,
    'Vessels_-_Dotted': 29
}


def replace_values(x):
    return torch.where(x >= 0.5, torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))

def busar_mascaras_conceito_dado_ID(id_imagem, pasta_mascaras, novo_tamanho=(7, 7)):
    # List to store the names of the concepts.
    conceitos = []
    # List to store resized images as tensors.
    imagens_tensores = []

    # Transformation to convert the image into a tensor
    transformacao = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(replace_values)
    ])

    # Iterating over the files in the folder.
    for arquivo in os.listdir(pasta_mascaras):
        # Check if the file starts with the image ID.
        if arquivo.startswith(id_imagem):
            #Extract the concept from the file name.
            conceito = arquivo[len(id_imagem) + 1:-4]  # Remove the '.jpg' extension.
            conceitos.append(conceito)

            # Full path to the mask file
            caminho_mascara = os.path.join(pasta_mascaras, arquivo)

            # Load the image using PIL
            imagem = Image.open(caminho_mascara).convert("L")  # "L" for grayscale image

            # Resize the image.
            imagem_redimensionada = transformacao(imagem.resize(novo_tamanho))

            # Add resized image to the list.
            imagens_tensores.append(imagem_redimensionada)

    return conceitos, imagens_tensores





def concept_eval(model, dataloader, size=(7,7)):
    print('\n\ttest')    
    model.eval()
    MIC3_masksfolder_path=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224_sum_across_concept_agree3"
    matriz_conceitos=torch.zeros(30, int(model.module.num_prototypes)).cuda()
    matriz_distancias=torch.zeros(30, int(model.module.num_prototypes)).cuda()


    for i, (image, label,ID,mask) in enumerate(dataloader):                
        input = image.cuda()
        num_imagens_ou_num_mascaras=input.shape[0]
        dimensao_mapa_latente=size[0]

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            x = model.module.conv_features(input)
            batch_size, D, P, P_same=x.shape
            distances=model.module.prototype_distances(input)
            for i in range(num_imagens_ou_num_mascaras):
                conceitos_da_imagem_i,mascaras_MIC3_da_imagem_i=busar_mascaras_conceito_dado_ID(ID[i],MIC3_masksfolder_path,novo_tamanho=size)
                for a in range(dimensao_mapa_latente):
                    for b in range(dimensao_mapa_latente):
                        distancias_patch_to_prototypes=distances[i,:,a,b]
                        idx_do_prototipo_mais_perto = torch.argmin(distancias_patch_to_prototypes)
                        for c in range(len(conceitos_da_imagem_i)):
                            numero_do_conceito = conceitos_numeros[conceitos_da_imagem_i[c]]
                            mic3_mascara_nao_invertida=mascaras_MIC3_da_imagem_i[c]
                            if(mic3_mascara_nao_invertida[0,a,b]==1):
                                matriz_conceitos[numero_do_conceito,idx_do_prototipo_mais_perto]+=1
                                matriz_distancias[numero_do_conceito,idx_do_prototipo_mais_perto]+=distancias_patch_to_prototypes[idx_do_prototipo_mais_perto]
    
    print(matriz_conceitos)
    # Division with Zero Division Handling
    matriz_distancias_medias = torch.where(matriz_conceitos != 0,matriz_distancias / matriz_conceitos, torch.tensor(0., device='cuda:0'))
    print(matriz_distancias_medias)

    # Convert to lists
    conceitos = list(conceitos_numeros.keys())
    prototipos = [f'Prototype {i}' for i in range(matriz_conceitos.shape[1])]

    # Create DataFrames
    df_conceitos = pd.DataFrame(matriz_conceitos.cpu().numpy(), columns=prototipos, index=conceitos)
    df_distancias = pd.DataFrame(matriz_distancias_medias.cpu().numpy(), columns=prototipos, index=conceitos)

    # Show DataFrames
    print("Concept Matrix:")
    print(df_conceitos)

    print("\nMean Distance Matrix:")
    print(df_distancias)
                                
                         
if __name__ == '__main__':
    import os
    import matplotlib
    import numpy as np
    import torch
    import torch.utils.data
    import torchvision.transforms as transforms
    matplotlib.use('TkAgg')  
    import random
    from Dataset import ISIC2019_Dataset
    from PIL import Image
    from settings import random_seed_number
    import torch.nn.functional as F
    from settings import online_augmentation
    import pandas as pd


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True
    
    print("Use only for the binary problem Melanoma Vs Nevus and when EDEASD images were used during training. That is, in the situation where the ISIC 2019 training set is combined with the EDEASD images.\n")
    train_push_dir = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224_sum_all_concept_agree3\images_for_this_masks\MEL_NV"
    load_model_path=r"C:\ACP_BM_Problem\NC2\resnet18\run1\20_2push0.8295.pth"
    size=(7,7)# Dimension of the maps that are the input of the prototype layer, i.e the output of convolution layers. Example [BATCH_SIZE,D,P,P]. So you should put size=(P,P). Only VGG16 has (14,14) the others is (7,7)

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    num_classes=ppnet_multi.module.num_classes
    if(num_classes!=2):
        print("It can only be used for the binary problem Melanoma vs Nevus.\n")
        exit(0)
    num_prototypes=ppnet_multi.module.num_prototypes
    
    train_dataset = ISIC2019_Dataset(image_dir=train_push_dir, is_train=True,number_classes=num_classes,augmentation=online_augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=250, shuffle=True,
        num_workers=4, pin_memory=False)


    concept_eval(model=ppnet_multi, dataloader=train_loader,size=size)

                         