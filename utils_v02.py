import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.knowledge_graph import FB15k237Dataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import RelGraphConv
from dataset import FoodDataset
from dgl.data.knowledge_graph import WN18Dataset
import random
from collections import defaultdict

def precision_k(emb, class_matrix, ingredient_name_similar_matrix, allergy_indices, allergy_idx, k):
    with torch.no_grad():

        emb = F.normalize(emb, p=2, dim=1)
        
        cosine_sim = torch.matmul(emb, emb.T)
        mask = torch.eye(cosine_sim.size(0), dtype=torch.bool, device=cosine_sim.device)

        cosine_sim.masked_fill_(mask, float('-inf'))
        num_nodes = emb.shape[0]
        reciprocal_ranks = []

        ingredient_name_similar_ranks = []

        allergy_ranks = []

        allergy_ratio = 0

        scaling_factor_class = sum(1.0 / i for i in range(1, k + 1))
        # scaling_factor_ingredient = sum(i for i in range(1, k + 1))
        
        for anchor_idx in range(len(allergy_indices)):
                anchor_index = allergy_indices[anchor_idx]

                _, top_k_indices = torch.topk(cosine_sim[anchor_index], k, largest=True)
                # top_k_indices = top_k_indices[1:]  # anchor 제외

                anchor_class_vector = class_matrix[anchor_index] # anchor의 class
            
                top_k_class_vectors = class_matrix[top_k_indices] # candiate의 class
            
                same_class = (top_k_class_vectors * anchor_class_vector).sum(dim=1) > 0
                ranks = torch.arange(1, k + 1).float()

            

                reciprocal_ranks_values = same_class.float() / ranks
            
                reciprocal_rank = reciprocal_ranks_values.sum().item() / scaling_factor_class
                reciprocal_ranks.append(reciprocal_rank)

            #
            
                ingredient_name_similar_rank = sum(ingredient_name_similar_matrix[anchor_index, top_k_indices] / ranks).item() / scaling_factor_class
                ingredient_name_similar_ranks.append(ingredient_name_similar_rank)

            #

                top_k_allergy_vectors = torch.isin(top_k_indices, allergy_indices)
                allergy_ranks_values = top_k_allergy_vectors.float() / ranks
                allergy_rank = allergy_ranks_values.sum().item() / scaling_factor_class
                allergy_ranks.append(allergy_rank)


                allergy_ratio += sum(top_k_allergy_vectors) / k

        avg_reciprocal_rank = sum(reciprocal_ranks) / len(allergy_indices)
        avg_ingredient_name_similar_rank = sum(ingredient_name_similar_ranks) / len(allergy_indices)
        avg_allergy_rank = sum(allergy_ranks) / len(allergy_indices)
        avg_allergy_ratio = allergy_ratio / len(allergy_ranks)


    return avg_reciprocal_rank, avg_ingredient_name_similar_rank, avg_allergy_rank, avg_allergy_ratio


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def nt_xent_loss(anchors, positives, negatives, temperature=0.5):
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)
    
    num_positives = anchors.shape[0]
    num_negatives = negatives.shape[0]
    
    pos_sim = torch.matmul(anchors, positives.T) / temperature
    neg_sim = torch.matmul(anchors, negatives.T) / temperature

    
    all_sim = torch.cat([pos_sim, neg_sim], dim=1)
    
    labels = torch.zeros(num_positives, dtype=torch.long, device=anchors.device)

    loss = F.cross_entropy(all_sim, labels)
    
    return loss

