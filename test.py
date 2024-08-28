import torch
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.knowledge_graph import FB15k237Dataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import RelGraphConv
from dgl.data.knowledge_graph import WN18Dataset
import random
from collections import defaultdict
import utils
from torch.utils.data import DataLoader, Dataset
import utils_v02

def get_allergy_dict(allergy_matrix):
    allergy_names = ['난류', '우유', '대두', '땅콩+견과류', '밀', 
                     '난류, 우유', '난류, 대두', '난류, 땅콩+견과류', '난류, 밀', '우유, 대두', '우유, 땅콩+견과류', '우유, 밀', '대두, 땅콩+견과류', '대두, 밀', '땅콩+견과류, 밀', 
                     '난류, 우유, 대두', '난류, 우유, 땅콩+견과류', '난류, 우유, 밀', '난류, 대두, 땅콩+견과류', '난류, 대두, 밀', '난류, 땅콩+견과류, 밀', '우유, 대두, 땅콩+견과류', '우유, 대두, 밀', '우유, 땅콩+견과류, 밀', '대두, 땅콩+견과류, 밀', 
                     '난류, 우유, 대두, 땅콩+견과류', '난류, 우유, 대두, 밀', '난류, 우유, 땅콩+견과류, 밀', '난류, 대두, 땅콩+견과류, 밀', '우유, 대두, 땅콩+견과류, 밀', 
                     '난류, 우유, 대두, 땅콩+견과류, 밀']
    
    # allergy_names = ['난류', '우유', '대두', '땅콩+견과류', '밀', 
    #                  '난류, 우유', '난류, 대두', '난류, 땅콩+견과류', '난류, 밀', '우유, 대두', '우유, 밀', '대두, 땅콩+견과류', '대두, 밀', '땅콩+견과류, 밀', 
    #                  '난류, 우유, 대두', '난류, 우유, 밀', '난류, 대두, 땅콩+견과류', '난류, 대두, 밀', '난류, 땅콩+견과류, 밀', '우유, 대두, 밀', '대두, 땅콩+견과류, 밀', 
    #                  '난류, 우유, 대두, 밀', '난류, 대두, 땅콩+견과류, 밀'
    #                  ]

    basic_allergies = ['난류', '우유', '대두', '땅콩+견과류', '밀']

    allergy_dict = {}

    for allergy_combination in allergy_names:
        allergies = allergy_combination.split(', ')
        
        allergy_mask = torch.ones(allergy_matrix.size(0), dtype=torch.bool, device=allergy_matrix.device)
        for allergy in allergies:
            index = basic_allergies.index(allergy)
            allergy_mask &= (allergy_matrix[:, index] == 1)
        
        allergy_indices = torch.nonzero(allergy_mask).squeeze()

        allergy_dict[allergy_combination] = allergy_indices

    return allergy_dict

if __name__ == "__main__":
    utils.set_random_seed(42)  # fixed random seed 

    class_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/class_matrix.pt')
    ingredient_name_similar_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/ingredient_name_similar_matrix.pt')
    
    allergy_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/allergy_matrix.pt')
    allergy_dict = get_allergy_dict(allergy_matrix)
    allergy_names = list(allergy_dict.keys())
    allergy_num = len(allergy_dict.keys())

    # embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method1/embedding_2000.pt') # subgraph

    # embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method2/v0.1/embedding_100.pt') # edge drop


    embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method2/v0.4/embedding_300.pt')
    # embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method2/baseline/embedding_300.pt')

    allergy_layers = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/layer_v0.2/v0.1/allergy_layers.pt')
    
    torch.save(embed, 'embed.pt')


    # layer_embed = allergy_layers[4](embed)
    # normalized_embeddings = F.normalize(layer_embed, p=2, dim=1)
    # cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)

    # mask = torch.eye(cosine_similarities.size(0), dtype=torch.bool)
    # cosine_similarities.masked_fill_(mask, float(0))

    # # cosine_similarities = (cosine_similarities + 1) / 2
    # # cosine_similarities = F.softmax(cosine_similarities, dim=1)


    # min_values = cosine_similarities.min(dim=1, keepdim=True)[0]
    # max_values = cosine_similarities.max(dim=1, keepdim=True)[0]
    # cosine_similarities = (cosine_similarities - min_values) / (max_values - min_values)

    # mask = torch.eye(cosine_similarities.size(0), dtype=torch.bool)
    # cosine_similarities.masked_fill_(mask, float(0))

    # # print(cosine_similarities)
    # print(torch.topk(cosine_similarities, 10, dim=1, largest=True))


    # print(normalized_similarity)
    # torch.save(normalized_similarity, '밀.pt')



    score = 0

    idx = 0

    # for allergy_idx in range(allergy_num):
    #     # layer_embed = embed
    #     # layer_embed = allergy_layers[allergy_idx](embed)

    #     allergy_name = allergy_names[allergy_idx]
    #     allergy_indices = allergy_dict[allergy_name]

    #     if allergy_name in ['우유, 땅콩+견과류', '난류, 우유, 땅콩+견과류', '우유, 대두, 땅콩+견과류', '우유, 땅콩+견과류, 밀', '난류, 우유, 대두, 땅콩+견과류', '난류, 우유, 땅콩+견과류, 밀', '우유, 대두, 땅콩+견과류, 밀', '난류, 우유, 대두, 땅콩+견과류, 밀']:
    #         layer_embed = embed
            

    #     else:
    #         layer_embed = allergy_layers[idx](embed)
    #         idx += 1

    #         normalized_embeddings = F.normalize(layer_embed, p=2, dim=1)
    #         cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)
    #         torch.save(cosine_similarities, f'{allergy_name}.pt')



    #     class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils_v02.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_indices.cpu(), allergy_idx, k=1)

    #     print(f'{allergy_name} class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')
    #     score += allergy_rank

    # avg_score = (score) / (allergy_num)
    # print(avg_score)



    for allergy_idx in range(allergy_num):
        # layer_embed = embed
        layer_embed = allergy_layers[allergy_idx](embed)

        allergy_name = allergy_names[allergy_idx]
        allergy_indices = allergy_dict[allergy_name]

        # normalized_embeddings = F.normalize(layer_embed, p=2, dim=1)
        # cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        # torch.save(cosine_similarities, f'{allergy_name}.pt')

        class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils_v02.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_indices.cpu(), allergy_idx, k=1)
        print(f'{allergy_name} class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')
        score += allergy_rank 

    avg_score = (score) / (allergy_num)
    print(avg_score)