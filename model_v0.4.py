import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
import utils
import torch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.dataloading import GraphDataLoader
import torch.cuda.amp as amp  


class EdgeDataset(Dataset):
    def __init__(self, edge_index, edge_type):
        self.edge_index = edge_index
        self.edge_type = edge_type

    def __len__(self):
        return self.edge_index.size(1)

    def __getitem__(self, idx):
        return self.edge_index[:, idx], self.edge_type[idx]

# class EdgeDataset(Dataset):
#     def __init__(self, edge_index, edge_type, num_nodes):
#         self.edge_index = edge_index
#         self.edge_type = edge_type
#         self.num_nodes = num_nodes

#     def __len__(self):
#         return self.edge_index.size(1)

#     def __getitem__(self, idx):
#         edge_idx = idx % self.num_nodes
#         relation_idx = idx // self.num_nodes
#         edge_index_for_relation = torch.nonzero(self.edge_type == relation_idx, as_tuple=False).squeeze(1)
#         selected_edge_idx = edge_index_for_relation[edge_idx]
        
#         return self.edge_index[:, selected_edge_idx], self.edge_type[selected_edge_idx]


class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, num_node, num_rels, hidden_channels=500, out_channels=500, ):
        super(RGCNEncoder, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_rels)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_rels)
        self.dropout = nn.Dropout(0.2)

        self.emb = nn.Embedding(num_node, hidden_channels)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.dropout(x)
        return x

class ContrastiveModel(torch.nn.Module):
    def __init__(self, encoder, num_rels):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.sigma = nn.Parameter(torch.ones(num_rels))

    def forward(self, edge_index, edge_type):
        embed = self.encoder(edge_index, edge_type)
        return embed

    def get_neg_pair_class(self, edge_class, negative_mask, k):

        anchor_samples = edge_class[0]
        pos_samples = edge_class[1]

        neg_samples = torch.zeros((anchor_samples.size(0), k), dtype=torch.long)

        for i, anchor_idx in enumerate(anchor_samples):
            true_indices = torch.nonzero(negative_mask[anchor_idx], as_tuple=False).squeeze(1)

            selected_indices = true_indices[torch.randperm(true_indices.size(0))[:k]]
            neg_samples[i] = selected_indices

        neg_samples = neg_samples.view(-1)

        return anchor_samples, pos_samples, neg_samples

    def nt_xent(self, anchor_embeds, positive_embeds, negative_embeds, temperature=0.5):
        anchor_embeds = F.normalize(anchor_embeds, dim=1)
        positive_embeds = F.normalize(positive_embeds, dim=1)
        negative_embeds = F.normalize(negative_embeds, dim=1)
 

        pos_sim = torch.sum(anchor_embeds * positive_embeds, dim=1) / temperature  # (num_pos,)
        pos_sim = pos_sim.unsqueeze(1)  # (num_pos, 1)

        neg_sim = torch.matmul(anchor_embeds, negative_embeds.T) / temperature  # (num_pos, num_neg)

        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # (num_pos, 1 + num_neg)

        labels = torch.zeros(pos_sim.size(0), dtype=torch.long, device='cuda:0')

        loss = F.cross_entropy(all_sim, labels)

        return loss

    def class_nt_xent(self, embeddings, class_pos_mask, edge_class, k):
        positive_mask = class_pos_mask
        negative_mask = ~positive_mask

        anchor_samples, pos_samples, neg_samples = self.get_neg_pair_class(edge_class, negative_mask, k)


        anchor_embeds = embeddings[anchor_samples]
        positive_embeds = embeddings[pos_samples]
        negative_embeds = embeddings[neg_samples]

        loss = self.nt_xent(anchor_embeds, positive_embeds, negative_embeds)

        return loss 
    

    def get_neg_pair_ingred(self, ingredient_name_similar_matrix, edge_ingred, top_k, ratio=10):
        anchor_samples = edge_ingred[0]
        pos_samples = edge_ingred[1]
        device = edge_ingred.device

        negative_samples = torch.zeros((anchor_samples.size(0), top_k), dtype=torch.long)

        for i, anchor_idx in enumerate(anchor_samples):
            similarity_scores = ingredient_name_similar_matrix[anchor_idx].to(device)
            bottom_indices = torch.argsort(similarity_scores).to(device)

            bottom_indices = bottom_indices[bottom_indices != anchor_idx] # 본인 제외
            bottom_100_indices = bottom_indices[:(top_k)*ratio]

            selected_neg_samples = random.sample(list(bottom_100_indices), top_k)  
            negative_samples[i] = torch.tensor(selected_neg_samples, dtype=torch.long)

        negative_samples = negative_samples.view(-1)


        return anchor_samples, pos_samples, negative_samples

    def ingred_nt_xent(self, embed, ingredient_name_similar_matrix, edge_ingred, k):

        anchor_samples, pos_samples, negative_samples = self.get_neg_pair_ingred(ingredient_name_similar_matrix, edge_ingred, k)


        anchor_embeds = embed[anchor_samples]

        num_nodes = anchor_embeds.size(0)  


        pos_embeds = embed[pos_samples]
        neg_embeds = embed[negative_samples]

        loss = self.nt_xent(anchor_embeds, pos_embeds, neg_embeds)

        return loss

    def get_loss(self, embed, ingredient_name_similar_matrix, class_pos_mask, edge_class, edge_ingred):
        total_loss = torch.tensor(0.0, device=embed.device)
            
        class_loss = self.class_nt_xent(embed, class_pos_mask, edge_class, 10)
        sigma1 = self.sigma[0]
        sigma_sq1 = sigma1**2
        total_loss += (1 / (2 * sigma_sq1)) * class_loss + torch.log(sigma1)

        ingred_loss = self.ingred_nt_xent(embed, ingredient_name_similar_matrix, edge_ingred, 10)
        sigma2 = self.sigma[1]
        sigma_sq2 = sigma2**2
        total_loss += (1 / (2 * sigma_sq2)) * ingred_loss + torch.log(sigma2)
        return total_loss

def get_edge(ingredient_name_similar_matrix, top_k=300):
    edge_index = []
    edge_weight = []

    for i in range(ingredient_name_similar_matrix.size(0)):

        similarity_scores = ingredient_name_similar_matrix[i].clone()
        similarity_scores[i] = -1  # 본인 인덱스의 값을 작은 값으로 설정

        topk_indices = torch.topk(ingredient_name_similar_matrix[i], top_k).indices
        topk_values = ingredient_name_similar_matrix[i][topk_indices]


        edge_index.append(torch.stack([torch.full_like(topk_indices, i), topk_indices]))
        edge_weight.append(topk_values)

    edge_index = torch.cat(edge_index, dim=1)
    edge_weight = torch.cat(edge_weight)

    return edge_index, edge_weight

def get_edge_class(positive_mask):
    num_node = positive_mask.size(1)
    anchor = torch.zeros(num_node, dtype=torch.long)
    selected_cols = torch.zeros(num_node, dtype=torch.long)

    for node_idx in range(num_node):
        true_indices = torch.nonzero(positive_mask[node_idx], as_tuple=False).squeeze(1)
        filtered_indices = true_indices[true_indices != node_idx] # 같은 class인데, 본인 제외
        selected_col = random.choice(filtered_indices)
        selected_cols[node_idx] = selected_col
        anchor[node_idx] = node_idx

    edge_index = torch.stack([anchor, selected_cols], dim=0)

    return edge_index

def get_ingred_edge(ingredient_name_similar_matrix, ratio=10):
    num_nodes = ingredient_name_similar_matrix.size(0)
    anchor = torch.zeros(num_node, dtype=torch.long)
    positive_samples = torch.zeros(num_nodes, dtype=torch.long)

    for node_idx in range(num_nodes):
        similarity_scores = ingredient_name_similar_matrix[node_idx]
        top_indices = torch.argsort(similarity_scores, descending=True)
        top_indices = top_indices[top_indices != node_idx] # 본인 제외
        top_k_indices = top_indices[:ratio]

        positive_sample = random.choice(top_k_indices)
        positive_samples[node_idx] = positive_sample
        anchor[node_idx] = node_idx
            
    edge_index = torch.stack([anchor, positive_samples], dim=0)
    edge_index = edge_index.T


    return edge_index

def create_dataloaders(edge_class_drop, edge_class_type, edge_ingred_drop, edge_ingred_type, batch_size):
    combined_edge_index = torch.cat([edge_class_drop, edge_ingred_drop], dim=1)
    combined_edge_type = torch.cat([edge_class_type, edge_ingred_type], dim=0)

    dataset = EdgeDataset(combined_edge_index, combined_edge_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# def create_dataloaders(edge_class_drop, edge_class_type, edge_ingred_drop, edge_ingred_type, num_nodes, batch_size):
#     combined_edge_index = torch.cat([edge_class_drop, edge_ingred_drop], dim=1)
#     combined_edge_type = torch.cat([edge_class_type, edge_ingred_type], dim=0)
    
#     dataset = EdgeDataset(combined_edge_index, combined_edge_type, num_nodes)
    
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return dataloader



def train(device, model_state_file, model, class_matrix, patience, ingredient_name_similar_matrix, allergy_matrix):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_precision = 0
    best_epoch = 0
    epochs_no_improve = 0
    allergy_num = allergy_matrix.size(1)
    epochs = 6000
    scaler = amp.GradScaler()
    num_nodes = class_matrix.size(0)
    embed_dim = model.encoder.emb.weight.size(1)  # 임베딩 차원 수
    best_embed = torch.zeros((num_nodes, embed_dim), device=device)
    batch_size = 10000


    class_pos_mask = torch.matmul(class_matrix, class_matrix.T) > 0

    for epoch in range(epochs):  # single graph batch
        model.train()

        edge_class_drop = class_drop_edge(class_matrix)
        edge_class_type = torch.zeros(edge_class_drop.size(1), dtype=torch.long, device=device) 

        # edge_class = get_edge_class(class_pos_mask)
        # edge_class_type = torch.zeros(edge_class.size(1), dtype=torch.long, device=device) 

        #

        edge_ingred_drop = ingred_drop_edge(ingredient_name_similar_matrix)
        edge_ingred_type = torch.ones(edge_ingred_drop.size(1), dtype=torch.long, device=device) 

        # edge_ingred = get_ingred_edge(ingredient_name_similar_matrix).T
        # edge_ingred_type = torch.ones(edge_ingred.size(1), dtype=torch.long, device=device) 


        total_edge_index = torch.cat([edge_class_drop, edge_ingred_drop], dim=1).to(device)
        total_edge_type = torch.cat([edge_class_type, edge_ingred_type], dim=0).to(device)

        # dataset = EdgeDataset(total_edge_index, total_edge_type)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        embed = model(total_edge_index, total_edge_type)
        loss = model.get_loss(embed, ingredient_name_similar_matrix, class_pos_mask, edge_class_drop, edge_ingred_drop)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
        optimizer.step()
        print("Epoch {:04d} | Loss {:.4f}".format(epoch, loss))


        # total_loss = 0
        # for batch in dataloader:
        #     batch_edge_index, batch_edge_type = batch
        #     batch_edge_index = batch_edge_index.to(device)
        #     batch_edge_type = batch_edge_type.to(device)

        #     batch_edge_class = batch_edge_index[batch_edge_type == 0].T
        #     batch_edge_ingred = batch_edge_index[batch_edge_type == 1].T

        #     embed = model(batch_edge_index.T, batch_edge_type)

        #     loss = model.get_loss(embed, ingredient_name_similar_matrix, class_pos_mask, batch_edge_class, batch_edge_ingred)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
        #     optimizer.step()

        #     total_loss += loss
    
        # print("Epoch {:04d} | Loss {:.4f}".format(epoch, total_loss / len(dataloader)))



        # with amp.autocast():
        #     embed = model(edge_index, edge_type)
        #     loss = model.get_loss(embed, ingredient_name_similar_matrix, class_pos_mask, edge_class, edge_ingred)

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()  
        # scaler.step(optimizer)
        # scaler.update()  

        # print("Epoch {:04d} | Loss {:.4f}".format(epoch, loss.item()))
        
        if (epoch + 1) % 100 == 0:
            allergy_num = allergy_matrix.size(1)
            model = model.cpu()
            model.eval()

            embed = model(total_edge_index.cpu(), total_edge_type.cpu())

            score = 0

            for allergy_idx in range(allergy_num):
                class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils.precision_k(embed, class_matrix, ingredient_name_similar_matrix, allergy_matrix, allergy_idx, k=10)
                # allergy_list = ['난류', '우유', '대두', '땅콩', '밀']
                score += (class_rank + ingredient_rank) / 2

                print(f'Validation class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')

            avg_score = (score) / (allergy_num)

            if best_precision < avg_score:
                best_precision = avg_score
                best_epoch = epoch
                epochs_no_improve = 0
                best_embed = embed
                torch.save({"state_dict": model.state_dict(), "epoch": epoch},model_state_file,)
                torch.save(best_embed, f'best_embedding_{epoch+1}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            model = model.to(device)
            torch.save(embed, f'embedding_{epoch+1}.pt')

def class_drop_edge(class_matrix, drop_prob=0.0006):
    num_node = class_matrix.size(0)

    class_pos = torch.matmul(class_matrix, class_matrix.T)
    class_pos.fill_diagonal_(0)
    
    class_mask = torch.zeros_like(class_pos)
    for node_idx in range(num_node):
        classes = class_pos[node_idx]
        equal_class_node = torch.sum(classes)+1
        class_rate = equal_class_node / num_node

        random_variable = torch.bernoulli(torch.full((num_node,), drop_prob*(1/class_rate)))

        class_mask[node_idx] = random_variable


    dropped_class = class_pos * class_mask
    edge_index = torch.nonzero(dropped_class, as_tuple=False).T
    return edge_index

def ingred_drop_edge(ingredient_name_similar_matrix):
    num_node = ingredient_name_similar_matrix.size(0)

    ingred_mask = torch.zeros_like(ingredient_name_similar_matrix)
    for node_idx in range(num_node):
        ingred = ingredient_name_similar_matrix[node_idx]
        top_k_values, _ = torch.topk(ingred, 7)
        mean_top_k = torch.mean(top_k_values)

        transformed_ingred = sigmoid_transform(ingred, 100, mean_top_k)

        random_variable = torch.bernoulli(transformed_ingred)
        ingred_mask[node_idx] = random_variable

    dropped_ingred = ingred_mask * ingredient_name_similar_matrix

    edge_index = torch.nonzero(dropped_ingred, as_tuple=False).T

    return edge_index


def sigmoid_transform(x, a, b):
    return 1 / (1 + torch.exp(-a * (x - b)))


def normalize_matrix_l1(matrix):
    row_sums = matrix.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1

    normalized_matrix = matrix / row_sums
    return normalized_matrix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_random_seed(42)  # fixed random seed 
    class_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=1/class_matrix.pt')
    ingredient_name_similar_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=1/ingredient_name_similar_matrix.pt')
    allergy_matrix = torch.load('C:/Users/UNIST/Desktop/ours/v0.4/data_v0.2/rel=1/allergy_matrix.pt')

    # x = class_matrix.to(device).float()
    # edge_index = (ingredient_name_similar_matrix > 0.5).nonzero(as_tuple=False)
    # edge_weight = ingredient_name_similar_matrix[edge_index[0], edge_index[1]].to(device).float()

    num_node = class_matrix.size(0)

    class_positive = (torch.matmul(class_matrix, class_matrix.T) > 0).float()

    # print(ingredient_name_similar_matrix)

    # ingredient_name_similar_matrix = ingredient_name_similar_matrix * class_positive
    # print(ingredient_name_similar_matrix)

    # edge_index = (ingredient_name_similar_matrix > 0).nonzero(as_tuple=False).t()
    # print(edge_index.shape)

    # edge_index, _ = get_edge(ingredient_name_similar_matrix)

    # data = Data(x=class_positive, edge_index=edge_index).to(device)

    # in_channels = class_positive.size(1)
    in_channels = 500

    num_rels = 2

    encoder = RGCNEncoder(in_channels, num_node, num_rels).to(device)
    model = ContrastiveModel(encoder, num_rels).to(device)

    model_state_file = "C:/Users/UNIST/Desktop/ours/v0.4/model_state_embed.pth"

    train(device, model_state_file, model, class_matrix, 10, ingredient_name_similar_matrix, allergy_matrix)



    # print("Testing...")
    # checkpoint = torch.load(model_state_file)
    # model = model.cpu()  # test on CPU
    # model.eval()
    # model.load_state_dict(checkpoint["state_dict"])
    # embed = model(data.x.cpu(), data.edge_index.cpu())
    
    # allergy_num = allergy_matrix.size(1)

    # torch.save(embed, 'embedding.pt')
    # # torch.save(allergy_layers, 'allergy_layers.pt')

    # for allergy_idx in range(allergy_num):
    #     class_rank, ingredient_rank, allergy_rank, allergy_ratio = utils.precision_k(embed, class_matrix, ingredient_name_similar_matrix, allergy_matrix, allergy_idx, k=10)
    #     # allergy_list = ['난류', '우유', '대두', '땅콩', '밀']

    #     print(f'Best class_rank: {class_rank:.3f} ingredient_rank: {ingredient_rank:.3f} allergy_rank: {allergy_rank:.3f} allergy_ratio: {allergy_ratio:.3f}')


