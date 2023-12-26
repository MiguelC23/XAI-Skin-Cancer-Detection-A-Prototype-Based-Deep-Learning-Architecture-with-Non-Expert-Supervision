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


def idx_to_coordinates(idx, map_size):
    b = idx % map_size
    a = idx // map_size
    return a,b

def replace_values(x):
    return torch.where(x >= 0.5, torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))

def busar_mascaras_conceito_dado_ID(id_imagem, pasta_mascaras, novo_tamanho=(7, 7)):
    # List to store the names of the concepts
    conceitos = []
    # List to store resized images as tensors.
    imagens_tensores = []

    # Transformation to convert the image into a tensor
    transformacao = transforms.Compose([
        transforms.ToTensor()
        #transforms.Lambda(replace_values)
    ])

    # Iterating over the files in the folder.
    for arquivo in os.listdir(pasta_mascaras):
        # Check if the file begins with the image ID.
        if arquivo.startswith(id_imagem):
            # Extract the concept from the file name.
            conceito = arquivo[len(id_imagem) + 1:-4]  # Remove the '.jpg' extension.
            conceitos.append(conceito)

            # Full path to the mask file
            caminho_mascara = os.path.join(pasta_mascaras, arquivo)

            # Load the image using PIL.
            imagem = Image.open(caminho_mascara).convert("L")  #"L" for grayscale image

            # Resize the image.
            imagem_redimensionada = transformacao(imagem.resize(novo_tamanho))

            #"Add resized image to the list.
            imagens_tensores.append(imagem_redimensionada)

    return conceitos, imagens_tensores


def concepts_present(ID,a,b,size):
    MIC3_masksfolder_path=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224_sum_across_concept_agree3"
    conceitos_da_imagem_i,mascaras_MIC3_da_imagem_i=busar_mascaras_conceito_dado_ID(ID,MIC3_masksfolder_path,novo_tamanho=size)
    print(conceitos_da_imagem_i)
    for c in range(len(conceitos_da_imagem_i)):
        numero_do_conceito = conceitos_numeros[conceitos_da_imagem_i[c]]
        mic3_mascara_nao_invertida=mascaras_MIC3_da_imagem_i[c]
        if(mic3_mascara_nao_invertida[0,a,b]>0):
            print(mic3_mascara_nao_invertida[0,a,b])
            print(f'Concept number {numero_do_conceito}, i.e {conceitos_da_imagem_i[c]} is present with a value of {mic3_mascara_nao_invertida[0,a,b]*100:0.4f}%. \n')

                                
            
if __name__ == '__main__':
    import os
    import matplotlib
    import numpy as np
    import torch
    import torch.utils.data
    import torchvision.transforms as transforms
    matplotlib.use('TkAgg')  
    import random
    from PIL import Image
    from settings import random_seed_number

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(random_seed_number)
    torch.cuda.manual_seed(random_seed_number)
    np.random.seed(random_seed_number)
    random.seed(random_seed_number)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    PROTO_ISIC_ID_EDEASD='ISIC_0046495'
    PATCH_IDX=25 # from 0 to 48

    # Exemplo de uso
    map_size = 7
    a,b = idx_to_coordinates(PATCH_IDX, map_size)
    print(f'Para IDX {PATCH_IDX}, as coordenadas são: [{a},{b}]')
    concepts_present(PROTO_ISIC_ID_EDEASD,a,b,(map_size,map_size))








