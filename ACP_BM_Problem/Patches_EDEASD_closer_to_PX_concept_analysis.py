import os
import re
import pandas as pd

def extract_number_from_string(input_string):
    start_str = "nearest-"
    end_str = "_latent_patch"
    
    start_index = input_string.find(start_str) + len(start_str)
    end_index = input_string.find(end_str)
    
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None
    
    number_str = input_string[start_index:end_index]
    try:
        number = int(number_str)
        return number
    except ValueError:
        return None

def extract_proto_idx_from_filename(filename):
    # Definir a expressão regular correta para encontrar os números finais
    pattern = r'_(\d+)\.npy$'
    match = re.search(pattern, filename)

    if match:
        # Retorna o valor correspondente à expressão regular encontrada
        return int(match.group(1))
    else:
        # Caso não encontre nenhum número final, retorna None ou um valor padrão (como -1)
        return None

def extract_unique_ISIC_names(folder_path):
    # Lista para armazenar os nomes únicos
    unique_ISIC_names = []

    # Expressão regular para encontrar o padrão "ISIC_[ID]" nos nomes dos ficheiros
    pattern = re.compile(r'ISIC_\d+')

    # Percorrer todos os ficheiros na pasta
    numbers_order=[]
    protos_indices=[]
    for filename in os.listdir(folder_path):
        #print(filename)
        number = extract_number_from_string(filename)
        idx_proto=extract_proto_idx_from_filename(filename)
        # Encontrar o padrão "ISIC_[ID]" no nome do ficheiro
        match = pattern.search(filename)
        if match:
            #print(filename)
            #print(idx_proto)
            #print(number)
            #print("\n")
            # Extrair e guardar o nome encontrado na lista de nomes únicos
            unique_ISIC_names.append(match.group())
            numbers_order.append(number)
            protos_indices.append(idx_proto)


    return unique_ISIC_names,numbers_order,protos_indices

def count_unique_strings(strings_list):
    unique_strings = {}
    for string in strings_list:
        unique_strings[string] = unique_strings.get(string, 0) + 1

    return unique_strings

def indices_ordenados(lista):
    # Use a função enumerate para atribuir um índice a cada elemento da lista
    # Em seguida, ordene os índices com base nos valores correspondentes da lista
    indices_ordenados = sorted(range(len(lista)), key=lambda x: lista[x])
    return indices_ordenados


def print_parameters_for_ISIC_ids(csv_file, unique_ISIC_names):
    # Leitura do arquivo CSV
    df = pd.read_csv(csv_file)

    # Filtrar as linhas que correspondem aos IDs únicos
    filtered_rows = df[df['ISIC_id'].isin(unique_ISIC_names)]

    # Sort the DataFrame by the order of unique_ISIC_names
    filtered_rows = filtered_rows.sort_values(by='ISIC_id', key=lambda x: x.map({name: i for i, name in enumerate(unique_ISIC_names)}))

    concepts_list = []
    # Imprimir os parâmetros para cada ISIC_id
    for _, row in filtered_rows.iterrows():
        ISIC_id = row['ISIC_id']
        exemplar = row['exemplar']
        group = row['group']
        benign_malignant = row['benign_malignant']
        diagnosis = row['diagnosis']
        concepts_list.append(exemplar)
        print(f"ISIC_id: {ISIC_id}, exemplar: {exemplar}, group: {group}, benign_malignant: {benign_malignant}, diagnosis: {diagnosis}")

    result_dict = count_unique_strings(concepts_list)

    # Display the unique strings and their counts
    for string, count in result_dict.items():
        print(f"'{string}': {count}")


# Substituir 'caminho_da_pasta' pelo caminho real da pasta que contém os ficheiros
pasta = r"C:\1CP_BinaryProblem\NC2\resnet18\run1\20_0push0.7869_nearest_from_EDEASD\0"
nomes_ISIC_unicos,numbers_order,protos_indices = extract_unique_ISIC_names(pasta)
indices = indices_ordenados(numbers_order)
nomes_por_ordem_mais_ativado=[]
protos_indices_por_ordem=[]
for i in range(len(indices)):
    nomes_por_ordem_mais_ativado.append(nomes_ISIC_unicos[indices[i]])
    protos_indices_por_ordem.append(protos_indices[indices[i]])
print(nomes_por_ordem_mais_ativado)
print(protos_indices_por_ordem)


# Substituir 'caminho_do_arquivo_csv' pelo caminho real do arquivo CSV
caminho_do_arquivo_csv = r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\data\metadata_concepts.csv"
print_parameters_for_ISIC_ids(caminho_do_arquivo_csv, nomes_por_ordem_mais_ativado)


import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


def replace_values(x):
    return torch.where(x < 0.5, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))

# Transformação da imagem
mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(replace_values)
])

def resize_image(mask_path):
    # Carregar a imagem em escala de cinza
    mask = Image.open(mask_path).convert('L')
    
    # Aplicar a função replace_values
    mask_tensor = mask_transform(mask)
    #unique_values = torch.unique(mask_tensor)
    #print(unique_values)
    # Expandir as dimensões do tensor para que se torne um tensor de imagem com um canal
    mask_tensor = mask_tensor.unsqueeze(0)
    
    # Aplicar o AdaptiveMaxPool2d para redimensionar a imagem para 7x7
    adaptive_pool = nn.AdaptiveMaxPool2d((7, 7))
    
    resized_mask = adaptive_pool(mask_tensor)
    #resized_mask=F.adaptive_avg_pool2d(mask_tensor, (7, 7))
    flattened_tensor = resized_mask.view(-1) #shape [49]

    return flattened_tensor

def extrair_parte_desejada(nome_imagem):
    partes = nome_imagem.split('_')
    
    # Verificar se a imagem tem o formato correto
    if len(partes) >= 3:
        parte_desejada = '_'.join(partes[2:])
        # Remover o prefixo com os números e o caractere "_"
        parte_desejada = parte_desejada.split('_', 1)[1]
        return parte_desejada
    else:
        return None

def remover_extensao(filename):
    nome_base, extensao = os.path.splitext(filename)
    return nome_base

def extract_ISIC_concepts_all_doctors(folder_path,top_conceitos,protos_indices):
    # Lista para armazenar os nomes únicos
    
    n=len(top_conceitos)
    concepts_identified_in_each_proto=[[] for _ in range(n)]

    # Expressão regular para encontrar o padrão "ISIC_[ID]" nos nomes dos ficheiros
    #for i in range(len(top_conceitos)):
    for i in range(len(top_conceitos)):
        pattern = top_conceitos[i]
        concepts = []
        #print(pattern)
        #print(protos_indices[i])
        #print("\n")
        #print(pattern)
        for filename in os.listdir(folder_path):
            
            # Encontrar o padrão "ISIC_[ID]" no nome do ficheiro
            nome_base=remover_extensao(filename)
            #match = pattern.search(filename)
            if pattern in filename:
                #print(filename)
                path=folder_path+'\\'+ filename
                mask_flat=resize_image(path)
                #print(mask_flat[protos_indices[i]])
                #print(protos_indices[i])
                #print("\n")
                if(mask_flat[protos_indices[i]]==1):
                    #print(filename)
                    #print(number)
                    # Extrair e guardar o nome encontrado na lista de nomes únicos
                    parte_desejada = extrair_parte_desejada(nome_base)
                    concepts.append(parte_desejada)
        concepts_identified_in_each_proto[i].append(concepts)



    return concepts_identified_in_each_proto

# Exemplo de uso:
print("\n===================================================================\n")
caminho_masks=r"C:\Users\migue\OneDrive\Ambiente de Trabalho\archive_ISIC\masks_224"
concepts_top_conceitos_all_doctors=extract_ISIC_concepts_all_doctors(caminho_masks,nomes_por_ordem_mais_ativado,protos_indices_por_ordem)
#print(concepts_top_conceitos_all_doctors)
#exit(0)

def contar_strings_unicas(lista):
    contagem = {}
    
    for item in lista:
        if item in contagem:
            contagem[item] += 1
        else:
            contagem[item] = 1

    return contagem

list_of_dictionarys=[]
for i in range(len(nomes_por_ordem_mais_ativado)):
    print("\n===================")
    print("Top ",i)
    resultado = contar_strings_unicas(concepts_top_conceitos_all_doctors[i][0])
    list_of_dictionarys.append(resultado)
    #print(resultado)
    # Exibindo o resultado
    for string, quantidade in resultado.items():
        print(f"{string}: {quantidade} vezes")

def find_most_common_keys(dictionaries):
    key_counter = {}
    for dictionary in dictionaries:
        keys_seen = set()
        for key in dictionary.keys():
            if key not in keys_seen:
                key_counter[key] = key_counter.get(key, 0) + 1
                keys_seen.add(key)
    
    most_common_keys = []
    max_count = 0
    for key, count in key_counter.items():
        if count > max_count:
            most_common_keys = [key]
            max_count = count
        elif count == max_count:
            most_common_keys.append(key)
    
    return most_common_keys,max_count


empty_indices = [index for index, dictionary in enumerate(list_of_dictionarys) if len(dictionary) == 0]
print("Indices of empty dictionaries:", empty_indices)
Number_of_empty_dicts=len(empty_indices)

# Example usage:
# dictionaries is a list containing the 20 dictionaries
# For example:
# dictionaries = [{...}, {...}, ..., {...}]  # 20 dictionaries with their respective keys and values
print("\n===================")
print("Most common concepts ignoring the minimun requirement of 3 dermatologists.")
result,max_count = find_most_common_keys(list_of_dictionarys)
N=len(nomes_por_ordem_mais_ativado)
print("Taking into consideration only  ",N- Number_of_empty_dicts, " images closest to the evaluated prototype, that had concepts present, the most prevalent concept(s) are:\n",result)
print("Which appear in ",max_count, " out of ",N- Number_of_empty_dicts," images.\n\n")

lista_de_dicionarios_new=list_of_dictionarys
import copy
lista_just_concepts_where_3_doctors_agree=copy.deepcopy(list_of_dictionarys)

def manter_apenas_maior_key(dicionarios):
    for dicionario in dicionarios:
        if not dicionario:  # Verifica se o dicionário está vazio
            continue

        # Encontra a maior key e seu valor
        maior_key = max(dicionario, key=dicionario.get)
        maior_valor = dicionario[maior_key]

        # Limpa o dicionário, mantendo apenas a maior key e seu valor
        dicionario.clear()
        dicionario[maior_key] = maior_valor

def imprimir_lista_de_dicionarios(lista):
    i=0
    for dicionario in lista:
        print("\n===================")
        print("Top ",i)
        i+=1
        for chave, valor in dicionario.items():
            print(f"{chave}: {valor}")
        print()

print("\n\nConducting the same analysis, but focusing solely on the concept with the highest agreement among the doctors for each image.")
manter_apenas_maior_key(lista_de_dicionarios_new)
imprimir_lista_de_dicionarios(lista_de_dicionarios_new)
result,max_count = find_most_common_keys(lista_de_dicionarios_new)
N=len(nomes_por_ordem_mais_ativado)
print("Taking into consideration only  ",N- Number_of_empty_dicts, " images closest to the evaluated prototype, that had concepts present, the most prevalent concept(s) are:\n",result)
print("Which appear in ",max_count, " out of ",N- Number_of_empty_dicts," images.\n\n")


def manter_acima_de_tres(dicionarios):
    for dicionario in dicionarios:
        if not dicionario:  # Verifica se o dicionário está vazio
            continue

        # Filtra as chaves cujos valores são maiores ou iguais a 3
        chaves_acima_de_tres = {chave: valor for chave, valor in dicionario.items() if valor >= 3}

        # Limpa o dicionário, mantendo apenas as chaves com valor acima ou igual a 3
        dicionario.clear()
        dicionario.update(chaves_acima_de_tres)

print("\n\nConducting the same analysis, but focusing solely on the concepts with the agreement of at least 3 doctors for each image.")
#print(lista_just_concepts_where_3_doctors_agree)
manter_acima_de_tres(lista_just_concepts_where_3_doctors_agree)
imprimir_lista_de_dicionarios(lista_just_concepts_where_3_doctors_agree)
result,max_count = find_most_common_keys(lista_just_concepts_where_3_doctors_agree)
N=len(nomes_por_ordem_mais_ativado)
print("Taking into consideration only  ",N- Number_of_empty_dicts, " images closest to the evaluated prototype, that had concepts present, the most prevalent concept(s) are:\n",result)
print("Which appear in ",max_count, " out of ",N- Number_of_empty_dicts," images.\n\n")
