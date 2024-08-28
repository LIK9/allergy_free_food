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
import utils_v02
from torch.utils.data import DataLoader, Dataset

class EmbedDataset(Dataset):
    def __init__(self, embed, sim_matrix, allergy_matrix):
        self.embed = embed
        self.sim_matrix = sim_matrix
        self.allergy_matrix = allergy_matrix

    def __len__(self):
        return self.embed.size(0)

    def __getitem__(self, idx):
        return self.embed[idx], self.sim_matrix[idx], self.allergy_matrix[idx], idx

class AllergyLayer(nn.Module):
    def __init__(self, allergy_num, h_dim=500):
        super().__init__()

        self.allergy_num = allergy_num
        self.allergy_layers = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(self.allergy_num)])

        for layer in self.allergy_layers:
            nn.init.eye_(layer.weight)  
            nn.init.constant_(layer.bias, 0)

        self.sigma = nn.Parameter(torch.ones(allergy_num))  


    def forward(self, embed, specific_allergy_idx):
        # embed_clone = embed.clone()
        
        # allergy_embed = embed[allergy_mask]  
        # layer_embeds = self.allergy_layers[specific_allergy_idx](allergy_embed)

        # embed_clone[allergy_mask] = layer_embeds

        layer_embeds = self.allergy_layers[specific_allergy_idx](embed)

        return layer_embeds
    
    def nt_xent_loss(self, anchor_embeds, positive_embeds, negative_embeds, temperature=0.001):
        anchor_embeds = F.normalize(anchor_embeds, dim=1)
        positive_embeds = F.normalize(positive_embeds, dim=1)
        negative_embeds = F.normalize(negative_embeds, dim=1)

        pos_sim = torch.sum(anchor_embeds * positive_embeds, dim=1) / temperature  # (num_pos,)
        pos_sim = pos_sim.unsqueeze(1)  # (num_pos, 1)

        neg_sim = torch.matmul(anchor_embeds, negative_embeds.T) / temperature  # (num_pos, num_neg)

        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (num_pos, 1 + num_neg)

        labels = torch.zeros(pos_sim.size(0), dtype=torch.long, device=embed.device)

        loss = F.cross_entropy(all_sim, labels)

        return loss
    
    # def get_loss(self, allergy_idx, layer_embed, sim_without_allergy, allergy_matrix, k=10):
    #     sample_size = 10000
    #     num_node = layer_embed.size(0)
    #     candidate_num = sim_without_allergy.size(1)


    #     allergy_mask = allergy_matrix[:, allergy_idx] == 1

    #     allergy_indices = torch.nonzero(allergy_mask).squeeze()
    #     non_allergy_indices = torch.nonzero(~allergy_mask).squeeze()

    #     sampled_indices = torch.randint(0, num_node, (sample_size,), device='cuda:0')

    #     non_allergy_anchor = sampled_indices[torch.isin(sampled_indices, non_allergy_indices)]
    #     allergy_anchor = sampled_indices[torch.isin(sampled_indices, allergy_indices)]
        
    #     non_allergy_embeds = layer_embed[non_allergy_anchor]
    #     allergy_embeds = layer_embed[allergy_anchor]

    #     # anchor

    #     non_allergy_candidates = sim_without_allergy[non_allergy_anchor]
    #     non_allergy_random_indices = torch.randint(0, candidate_num, (non_allergy_candidates.size(0),))
    #     non_allergy_pos_indices = non_allergy_candidates[torch.arange(non_allergy_candidates.size(0)), non_allergy_random_indices]
    #     non_allergy_positive_embeds = layer_embed[non_allergy_pos_indices]



    #     allergy_positive_samples = []
    #     for anchor in allergy_anchor:
    #         possible_indices = allergy_indices[allergy_indices != anchor]
    #         random_sample = possible_indices[torch.randint(0, len(possible_indices), (1,))]
    #         allergy_positive_samples.append(random_sample.item())

    #     allergy_positive_samples = torch.tensor(allergy_positive_samples, device='cuda:0')
    #     allergy_positive_embeds = layer_embed[allergy_positive_samples]


    #     # positive

    #     non_allergy_anchor_num = non_allergy_anchor.size(0)
    #     allergy_anchor_num = allergy_anchor.size(0)

    #     non_allergy_negative_sample = allergy_indices[torch.randint(0, allergy_indices.size(0), (non_allergy_anchor_num*k,), device=allergy_indices.device)]
    #     non_allergy_negative_embeds = layer_embed[non_allergy_negative_sample]

    #     allergy_negative_sample = non_allergy_indices[torch.randint(0, non_allergy_indices.size(0), (allergy_anchor_num*k,), device=allergy_indices.device)]
    #     allergy_negative_embeds = layer_embed[allergy_negative_sample]

    #     # # negative


    #     non_allergy_loss = self.nt_xent_loss(non_allergy_embeds, non_allergy_positive_embeds, non_allergy_negative_embeds)
    #     allergy_loss = self.nt_xent_loss(allergy_embeds, allergy_positive_embeds, allergy_negative_embeds)


    #     loss = non_allergy_loss + allergy_loss

    #     return loss 
    
    # def get_loss(self, allergy_idx, layer_embed, allergy_matrix, temperature=0.5, k=10):
    #     sample_size = 10000
    #     num_node = layer_embed.size(0)
        
    #     allergy_mask = allergy_matrix[:, allergy_idx] == 1

    #     allergy_indices = torch.nonzero(allergy_mask).squeeze()

    #     allergy_num = allergy_indices.size(0)

    #     sampled_indices = torch.randint(0, allergy_num, (sample_size,), device='cuda:0')

    #     anchor_indices = allergy_indices[sampled_indices]

    #     anchor_embeds = layer_embed[anchor_indices]


    #     # anchor

    #     candidate_num = sim_without_allergy.size(1)
    #     positive_candidates = sim_without_allergy[anchor_indices]
    #     random_indices = torch.randint(0, candidate_num, (sample_size,))
    #     pos_indices = positive_candidates[torch.arange(sample_size), random_indices]
    #     positive_embeds = layer_embed[pos_indices]

    #     # positive

    #     negative_samples = torch.empty((sample_size, 10), dtype=torch.long, device='cuda:0')
    #     for i in range(sample_size):
    #         anchor_index = anchor_indices[i]
    #         neagtive_candiate = allergy_indices[allergy_indices != anchor_index]
    #         random_indices = torch.randperm(neagtive_candiate.size(0))[:10]
    #         negative_samples[i] = neagtive_candiate[random_indices]

    #     negative_samples = negative_samples.view(-1)
    #     negative_embeds = layer_embed[negative_samples]

    #     # negative

    #     loss = self.nt_xent_loss(anchor_embeds, positive_embeds, negative_embeds)

    #     return loss

    # def get_loss(self, allergy_idx, layer_embed, allergy_matrix):
    #         num_node = layer_embed.size(0)
            
    #         allergy_mask = allergy_matrix[:, allergy_idx] == 1

    #         allergy_indices = torch.nonzero(allergy_mask).squeeze()
    #         non_allergy_indices = torch.nonzero(~allergy_mask).squeeze()

    #         normalized_embeddings = F.normalize(layer_embed, p=2, dim=1)
    #         cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)


    #         allergy_num = allergy_indices.size(0)

    #         anchor_embeds = layer_embed[allergy_indices]


    #         # anchor

    #         # candidate_num = sim_without_allergy.size(1)
    #         # positive_candidates = sim_without_allergy[anchor_indices]
    #         # random_indices = torch.randint(0, candidate_num, (sample_size,))
    #         # pos_indices = positive_candidates[torch.arange(sample_size), random_indices]
    #         # positive_embeds = layer_embed[pos_indices]

    #         positive_indices = self.get_pos_pair(cosine_similarities, allergy_indices, non_allergy_indices)
    #         positive_embeds = layer_embed[positive_indices]

    #         # positive

    #         # negative_samples = torch.empty((allergy_num, (allergy_num - 1)), dtype=torch.long, device='cuda:0')    
    #         # for i in range(allergy_num):
    #         #     negative_sample = torch.cat((allergy_indices[:i], allergy_indices[i+1:]))
    #         #     negative_samples[i] = negative_sample
            
    #         # negative_samples = negative_samples.view(-1)
    #         # negative_embeds = layer_embed[negative_samples]

    #         negative_indices = self.get_neg_pair(cosine_similarities, allergy_indices)
    #         negative_embeds = layer_embed[negative_indices]

    #         # negative

    #         loss = self.nt_xent_loss(anchor_embeds, positive_embeds, negative_embeds)

    #         return loss
    
    def get_loss(self, layer_embed, allergy_indices):
            num_node = layer_embed.size(0)
            sample_size = 1000

            all_indices = torch.arange(num_node, device=layer_embed.device)
            allergy_indices = allergy_indices
            non_allergy_indices = torch.tensor(list(set(all_indices.cpu().numpy()) - set(allergy_indices.cpu().numpy())), device=layer_embed.device)


            perm = torch.randint(0, allergy_indices.size(0), (sample_size,))
            anchor_indices = allergy_indices[perm]

            normalized_embeddings = F.normalize(layer_embed, p=2, dim=1)
            cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)


            anchor_embeds = layer_embed[anchor_indices]


            # anchor

            positive_indices = self.get_pos_pair(cosine_similarities, anchor_indices, non_allergy_indices)
            positive_embeds = layer_embed[positive_indices]

            # positive

            negative_indices = self.get_neg_pair(cosine_similarities, anchor_indices, allergy_indices)
            negative_embeds = layer_embed[negative_indices]

            # negative

            loss = self.nt_xent_loss(anchor_embeds, positive_embeds, negative_embeds)

            return loss

    def get_pos_pair(self, cosine_similarities, anchor_indices, non_allergy_indices):
        allergy_to_non_allergy_sim = cosine_similarities[anchor_indices][:, non_allergy_indices]
        

        most_similar_non_allergy_indices = torch.argmax(allergy_to_non_allergy_sim, dim=1)

        positive_samples = non_allergy_indices[most_similar_non_allergy_indices]

        # positive_samples = torch.empty((allergy_num), dtype=torch.long, device='cuda:0')

        # for i in range(most_similar_non_allergy_indices.size(0)):
        #     positive_sample = non_allergy_indices[most_similar_non_allergy_indices[i]]
        #     print(positive_sample)
        #     positive_samples[i] = positive_sample
        return positive_samples
    
    def get_neg_pair(self, cosine_similarities, anchor_indices, allergy_indices, top_k = 10):
        allergy_to_allergy_sim = cosine_similarities[anchor_indices][:, anchor_indices]
        mask = torch.eye(allergy_to_allergy_sim.size(0), dtype=torch.bool, device=allergy_to_allergy_sim.device)

        allergy_to_allergy_sim.masked_fill_(mask, float('-inf'))

        top10_similar_indices = torch.topk(allergy_to_allergy_sim, top_k, dim=1).indices
        negative_samples = anchor_indices[top10_similar_indices].view(-1)


        return negative_samples
    
    # def get_neg_pair(self, cosine_similarities, anchor_indices, allergy_indices, top_k = 5): # true neg
    #     allergy_to_allergy_sim = cosine_similarities[anchor_indices][:, allergy_indices]

    #     mask = allergy_indices.unsqueeze(0) == anchor_indices.unsqueeze(1)

    #     allergy_to_allergy_sim.masked_fill_(mask, float('-inf'))

    #     top10_similar_indices = torch.topk(allergy_to_allergy_sim, top_k, dim=1).indices
    #     negative_samples = allergy_indices[top10_similar_indices].view(-1)

    #     return negative_samples


        # positive_candidates = top_k_indices[anchor_indices]

        # random_indices = torch.randint(0, top_k, (sample_size,))
        # pos_indices = positive_candidates[torch.arange(sample_size), random_indices]

        # final_pos_indices = non_allergy_indices[pos_indices]
        # print(final_pos_indices)

    # def total_loss(self, loss_array):
    #     total_loss = torch.tensor(0.0, device=loss_array.device)

    #     # for allergy_idx in range(self.allergy_num):
    #     #     sigma_sq = self.sigma[allergy_idx] ** 2

    #     #     loss = loss_array[allergy_idx]

    #     #     total_loss += (1 / (2 * sigma_sq)) * loss + torch.log(self.sigma[allergy_idx])

    #     sigma_softmax = F.softmax(self.sigma, dim=0)
    #     print(sigma_softmax)

    #     for allergy_idx in range(self.allergy_num):

    #         loss = loss_array[allergy_idx]

    #         total_loss += loss * (1 - sigma_softmax[allergy_idx])

    #     # print(self.sigma)

    #     return total_loss


# def train(model, device, allergy_num, embed, allergy_matrix, class_matrix, ingredient_name_similar_matrix, model_state_file, epochs=35000, per_valid=500):
#     best_score = 1

#     for epoch in range(epochs):
#         model.train()
        

#         total_loss = 0

#         for allergy_idx in range(allergy_num):
#             model.train()
#             optimizer = torch.optim.Adam(model.allergy_layers[allergy_idx].parameters(), lr=1e-4)
#             layer_embed = model(embed, allergy_idx)

#             loss = model.get_loss(allergy_idx, layer_embed, allergy_matrix)
            
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             nn.utils.clip_grad_norm_(model.allergy_layers[allergy_idx].parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss += loss.item()

#         # avg_loss = total_loss / allergy_num

#         print("Epoch {:04d} | Loss {:.4f}".format(epoch, total_loss))

#         if (epoch + 1) % 500 == 0:
#             model.eval()
#             model = model.cpu()

#             score = 0

#             for allergy_idx in range(allergy_num):
#                 layer_embed = model(embed.cpu(), allergy_idx)
#                 class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_matrix.cpu(), allergy_idx, k=1)
#                 score += allergy_rank + allergy_ratio

#                 print(f'Validation class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')

            
#             avg_score = (score) / (2*allergy_num)
#             print(avg_score)

#             if avg_score < best_score:
#                 best_score = avg_score
#                 torch.save({"state_dict": model.state_dict(), "epoch": epoch},model_state_file,)

#             model = model.to(device)

def train(model, device, allergy_num, embed, allergy_dict, class_matrix, ingredient_name_similar_matrix, model_state_file, epochs=60000, per_valid=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_score = 1
    allergy_names = list(allergy_dict.keys())

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        # loss_array = torch.zeros(allergy_num).to(device=embed.device)

        for allergy_idx in range(allergy_num):
            allergy_name = allergy_names[allergy_idx]
            allergy_indices = allergy_dict[allergy_name]

            layer_embed = model(embed, allergy_idx)

            # sim_without_allergy = sim_matrix[allergy_idx]

            loss = model.get_loss(layer_embed, allergy_indices)

            sigma = model.sigma[allergy_idx]
            sigma_sq = sigma**2
            total_loss += (1 / (2 * sigma_sq)) * loss + torch.log(sigma)

            # total_loss += loss
            
            # loss_array[allergy_idx] = loss

        # total_loss = model.total_loss(loss_array)

        # avg_loss = total_loss / allergy_num

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
        optimizer.step()
        print("Epoch {:04d} | Loss {:.4f}".format(epoch, total_loss))

        if (epoch + 1) % 500 == 0:
            model.eval()
            model = model.cpu()

            score = 0

            for allergy_idx in range(allergy_num):
                layer_embed = model(embed.cpu(), allergy_idx)

                allergy_name = allergy_names[allergy_idx]
                allergy_indices = allergy_dict[allergy_name]

                class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils_v02.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_indices.cpu(), allergy_idx, k=1)
                score += allergy_rank + allergy_ratio

                print(f'Validation {allergy_name} class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')

            
            avg_score = (score) / (2*allergy_num)
            print(avg_score)

            if avg_score < best_score:
                best_score = avg_score
                torch.save({"state_dict": model.state_dict(), "epoch": epoch},model_state_file,)

            model = model.to(device)


# def find_top_k_similar_nodes(embeddings, allergy_matrix, k=1):

#     normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
#     cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
#     num_allergies = allergy_matrix.size(1)
#     num_nodes = embeddings.size(0)
#     similar_nodes_by_allergy = torch.zeros((num_allergies, num_nodes, k), dtype=torch.long, device=embeddings.device)

#     for allergy_idx in range(num_allergies):
#         print(allergy_idx)
#         for node_idx in range(num_nodes):
#             sorted_similarities, sorted_indices = torch.sort(cosine_similarities[node_idx], descending=True)
#             selected_indices = []
#             for idx in sorted_indices:
#                 if idx != node_idx and allergy_matrix[idx, allergy_idx] == 0:  # 알레르기가 없는 노드만 선택
#                     selected_indices.append(idx.item())
#                 if len(selected_indices) == k:
#                     break
#             similar_nodes_by_allergy[allergy_idx, node_idx, :] = torch.tensor(selected_indices, device=embeddings.device)

#     torch.save(similar_nodes_by_allergy, 'sim_without_allergy.pt')

def get_allergy_dict(allergy_matrix):
    # allergy_names = ['난류', '우유', '대두', '땅콩+견과류', '밀', 
    #                  '난류, 우유', '난류, 대두', '난류, 땅콩+견과류', '난류, 밀', '우유, 대두', '우유, 밀', '대두, 땅콩+견과류', '대두, 밀', '땅콩+견과류, 밀', 
    #                  '난류, 우유, 대두', '난류, 우유, 밀', '난류, 대두, 땅콩+견과류', '난류, 대두, 밀', '난류, 땅콩+견과류, 밀', '우유, 대두, 밀', '대두, 땅콩+견과류, 밀', 
    #                  '난류, 우유, 대두, 밀', '난류, 대두, 땅콩+견과류, 밀'
    #                  ]

    allergy_names = ['난류', '우유', '대두', '땅콩+견과류', '밀', 
                     '난류, 우유', '난류, 대두', '난류, 땅콩+견과류', '난류, 밀', '우유, 대두', '우유, 땅콩+견과류', '우유, 밀', '대두, 땅콩+견과류', '대두, 밀', '땅콩+견과류, 밀', 
                     '난류, 우유, 대두', '난류, 우유, 땅콩+견과류', '난류, 우유, 밀', '난류, 대두, 땅콩+견과류', '난류, 대두, 밀', '난류, 땅콩+견과류, 밀', '우유, 대두, 땅콩+견과류', '우유, 대두, 밀', '우유, 땅콩+견과류, 밀', '대두, 땅콩+견과류, 밀', 
                     '난류, 우유, 대두, 땅콩+견과류', '난류, 우유, 대두, 밀', '난류, 우유, 땅콩+견과류, 밀', '난류, 대두, 땅콩+견과류, 밀', '우유, 대두, 땅콩+견과류, 밀', 
                     '난류, 우유, 대두, 땅콩+견과류, 밀']

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
    utils_v02.set_random_seed(42)  # fixed random seed 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    allergy_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/allergy_matrix.pt').to(device)
    allergy_dict = get_allergy_dict(allergy_matrix)
    allergy_names = list(allergy_dict.keys())

    allergy_num = len(allergy_dict.keys())
    # allergy_num = 1

    # embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method2/v0.1/embedding_100.pt').to(device) # edge drop
    embed = torch.load(f'C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=2/embed_v0.4/method2/v0.4/embedding_300.pt').to(device)


    # find_top_k_similar_nodes(embed, allergy_matrix, 10)

    # sim_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=1/sim_without_allergy.pt').to(device)

    # dataset = EmbedDataset(embed, sim_matrix, allergy_matrix)

    # train_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

    model = AllergyLayer(allergy_num).to(device)

    class_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/class_matrix.pt')
    ingredient_name_similar_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/ingredient_name_similar_matrix.pt')

    model_state_file = "C:/Users/UNIST/Desktop/ours/v0.4/model_state_layer.pth"


    # allergy_matrix = allergy_matrix[:, 0].unsqueeze(-1)

    # sim_matrix = sim_matrix[0, :, :].unsqueeze(0)


    train(model, device, allergy_num, embed, allergy_dict, class_matrix, ingredient_name_similar_matrix, model_state_file)

    print('Testing...')

    checkpoint = torch.load(model_state_file)
    model = model.cpu()  # test on CPU
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])

    allergy_layers = model.allergy_layers
    torch.save(allergy_layers, 'C:/Users/UNIST/Desktop/ours/v0.4/allergy_layers.pt')

    for allergy_idx in range(allergy_num):
        layer_embed = model(embed.cpu(), allergy_idx)

        allergy_name = allergy_names[allergy_idx]
        allergy_indices = allergy_dict[allergy_name]

        class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils_v02.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_indices.cpu(), allergy_idx, k=1)
        print(f'Best {allergy_name} class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f} epoch : {checkpoint["epoch"]}')
    
    print('-------------------')
    
    for allergy_idx in range(allergy_num):
        layer_embed = model(embed.cpu(), allergy_idx)

        allergy_name = allergy_names[allergy_idx]
        allergy_indices = allergy_dict[allergy_name]

        class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils_v02.precision_k(layer_embed, class_matrix, ingredient_name_similar_matrix, allergy_indices.cpu(), allergy_idx, k=10)
        print(f'Best {allergy_name} class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f} epoch : {checkpoint["epoch"]}')