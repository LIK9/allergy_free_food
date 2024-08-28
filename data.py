import pandas as pd
import numpy as np
import csv
import random
from torch.utils.data import random_split
from scipy.sparse import csr_matrix, save_npz, load_npz
import torch
import re

def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_food_vocab(ingredient_DF):
    food_vocab = np.unique(ingredient_DF['name'])
    return food_vocab

def get_ingredient_vocab(ingredient_DF):
    ingredient_vocab = np.unique(ingredient_DF['ingredient'].values)
    return ingredient_vocab

def get_food_with_vocab(diets_DF, nutri_DF):
    diets_array = diets_DF.to_numpy()
    
    unique_foods = np.unique(diets_array)
    unique_foods = unique_foods[unique_foods != 'empty']

    snack_foods = np.unique(nutri_DF[nutri_DF['snack'] == 1]['name'])

    common_foods = np.intersect1d(unique_foods, snack_foods)

    nutri_foods = np.unique( nutri_DF['name'] )
    nutri_foods = nutri_foods[nutri_foods != 'empty']
    
    food_vocab = np.setdiff1d(nutri_foods, snack_foods)

    food_vocab = np.concatenate((food_vocab, common_foods))

    return np.unique(food_vocab)

def save_class_label(food_vocab, diets_DF):
    label_count_matrix = np.zeros((len(food_vocab), 5))

    for i in range(len(food_vocab)):
        food = food_vocab[i]
        print(f"class_label processing : {i / len(food_vocab) * 100:.2f}%") # process
        matches = np.where(diets_DF == food)
        for _, col in zip(matches[0], matches[1]):
            if col in [0, 3, 8]: # 밥
                label_count_matrix[i][0] += 1
            elif col in [1, 4, 9]: # 국
                label_count_matrix[i][1] += 1
            elif col in [2, 5, 10]: # 주찬
                label_count_matrix[i][2] += 1
            elif col in [6, 11]: # 부찬
                label_count_matrix[i][3] += 1
            else: # 김치
                label_count_matrix[i][4] += 1

    zero_rows = np.all(label_count_matrix == 0, axis=1)

    indices = np.where(zero_rows)[0]
    class_label = np.argmax(label_count_matrix, axis=1)
    
    class_label[indices] = 6

    np.save('C:/Users/UNIST/Desktop/ours/KG/data/food_class_label.npy', class_label)

def save_class_txt(class_label, food_vocab):
    class_0 = []
    class_1 = []
    class_2 = []
    class_3 = []
    class_4 = []
    class_5 = []

    for idx in range(len(food_vocab)):
        food = food_vocab[idx]
        class_ = class_label[idx]
        if class_ == 0: # 밥
            class_0.append(food)
        elif class_ == 1: # 국
            class_1.append(food)
        elif class_ == 2: # 주찬
            class_2.append(food)
        elif class_ == 3: # 부찬
            class_3.append(food) 
        elif class_ == 4: # 김치
            class_4.append(food)
        else: # 기타 
            class_5.append(food)

    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/밥.txt', 'w') as f:
        f.write('\n'.join(class_0))
    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/국.txt', 'w') as f:
        f.write('\n'.join(class_1))
    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/주찬.txt', 'w') as f:
        f.write('\n'.join(class_2))
    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/부찬.txt', 'w') as f:
        f.write('\n'.join(class_3))
    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/김치.txt', 'w') as f:
        f.write('\n'.join(class_4))
    with open('C:/Users/UNIST/Desktop/ours/KG/data/class/original/기타.txt', 'w') as f:
        f.write('\n'.join(class_5))

def get_class_label():
    class_label = np.load('C:/Users/UNIST/Desktop/ours/KG/data/food_class_label.npy')

    return class_label

def get_class_vector(class_label, food_vocab):
    class_0 = []
    class_1 = []
    class_2 = []
    class_3 = []
    class_4 = []
    class_5 = []

    for idx in range(len(food_vocab)):
        food = food_vocab[idx]
        class_ = class_label[idx]
        if class_ == 0: # 밥
            class_0.append(food)
        elif class_ == 1: # 국
            class_1.append(food)
        elif class_ == 2: # 주찬
            class_2.append(food)
        elif class_ == 3: # 부찬
            class_3.append(food) 
        elif class_ == 4: # 김치
            class_4.append(food)
        else:
            class_5.append(food)
    
    class_vector = []

    for food in class_0:
        for class_food in class_0:
            if food != class_food:
                class_vector.append(f'{food}\tclass_rice\t{class_food}')
    
    for food in class_1:
        for class_food in class_1:
            if food != class_food:
                class_vector.append(f'{food}\tclass_soup\t{class_food}')
    
    for food in class_2:
        for class_food in class_2:
            if food != class_food:
                class_vector.append(f'{food}\tclass_main_dish\t{class_food}')

    for food in class_3:
        for class_food in class_3:
            if food != class_food:
                class_vector.append(f'{food}\tclass_side_dish\t{class_food}')
    
    for food in class_4:
        for class_food in class_4:
            if food != class_food:
                class_vector.append(f'{food}\tclass_kimchi\t{class_food}')
    
    return np.unique(class_vector)

# def get_with_vector_index(food_vocab, diets_DF):
#     with_vector_index = []
#     food_vocab = list(food_vocab)
#     for i in range(len(food_vocab)):
#         food = food_vocab[i]
#         print(f"with_vector_index processing : {i / len(food_vocab) * 100:.2f}%") # process
#         matches = np.where(diets_DF == food)

#         for row, col in zip(matches[0], matches[1]):
#             if col <= 2:  # breakfast
#                 with_food_cols = np.array([0, 1, 2])
#             elif col <= 7:  # lunch
#                 with_food_cols = np.array([3, 4, 5, 6, 7])
#             else:  # dinner
#                 with_food_cols = np.array([8, 9, 10, 11, 12])
            
#             with_foods = diets_DF.iloc[row, with_food_cols]
            
#             unique_with_foods = np.unique(with_foods[with_foods != 'empty'])
            
#             for with_food in unique_with_foods:
#                 if food != with_food:
#                     food_index = food_vocab.index(food)
#                     with_food_index = food_vocab.index(with_food)
#                     # with_vector_index.append(f'{food_index}\t{with_food_index}')
#                     with_food_string = f"{food_index}\t{with_food_index}"
#                     with_vector_index.append(with_food_string)
                
#     return with_vector_index
    
# def get_adj_matrix(with_vector_index, food_vocab):
#     food_food_with_adj_matrix = np.zeros((len(food_vocab), len(food_vocab)), dtype=int)

#     for with_index in with_vector_index:
#         with_index = with_index.split('\t')

#         food_index = int(with_index[0])
#         with_food_index = int(with_index[1])
        
#         food_food_with_adj_matrix[food_index, with_food_index] = 1

#     food_food_with_adj_matrix = csr_matrix(food_food_with_adj_matrix)

#     return food_food_with_adj_matrix

def get_diet_matrix(diets_DF):
    diets_matrix = diets_DF.to_numpy()
    breakfast_matrix = diets_matrix[:, :3]
    lunch_matrix = diets_matrix[:, 3:8]
    dinner_matrix = diets_matrix[:, 8:]

    return breakfast_matrix, lunch_matrix, dinner_matrix

def get_meal_similar_vector_breakfast(breakfast_matrix, breakfast):
    meal_similar_vectors = []

    for r in range(breakfast_matrix.shape[0]): # breakfast
        row = breakfast_matrix[r]
        for i in range(len(row)):
            food = row[i]
            if food != 'empty':
                if i == 0:
                    meal_similar_vector = np.where( (breakfast_matrix[:, 1] == row[1]) & (breakfast_matrix[:, 2] == row[2]) )
                    for with_food_row in meal_similar_vector[0]:
                        with_food = breakfast_matrix[with_food_row][0]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{breakfast}\t{with_food}')
                elif i == 1:
                    meal_similar_vector = np.where( (breakfast_matrix[:, 0] == row[0]) & (breakfast_matrix[:, 2] == row[2]) )
                    for with_food_row in meal_similar_vector[0]:
                        with_food = breakfast_matrix[with_food_row][1]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{breakfast}\t{with_food}')
                else:
                    meal_similar_vector = np.where( (breakfast_matrix[:, 0] == row[0]) & (breakfast_matrix[:, 1] == row[1]) )
                    for with_food_row in meal_similar_vector[0]:
                        with_food = breakfast_matrix[with_food_row][2]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{breakfast}\t{with_food}')

    return np.unique(meal_similar_vectors)

def get_meal_similar_vector_lunch_dinner(meal_matrix, lunch_or_dinner):
    meal_similar_vectors = []

    for r in range(meal_matrix.shape[0]): 
        row = meal_matrix[r]
        
        for i in range(len(row)):
            food = row[i]
            if food != 'empty':
                if i == 0:
                    # meal_similar_vector = np.where(((meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3])))
                    meal_similar_vector = np.where( (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4]) )
                    
                    for with_food_row in meal_similar_vector[0]:
                        with_food = meal_matrix[with_food_row][0]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{lunch_or_dinner}\t{with_food}')

                elif i == 1:
                    # meal_similar_vector = np.where(((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3])) |
                    #                             ((meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])))

                    meal_similar_vector = np.where( (meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4]) )

                    for with_food_row in meal_similar_vector[0]:
                        with_food = meal_matrix[with_food_row][1]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{lunch_or_dinner}\t{with_food}')

                elif i == 2:
                    # meal_similar_vector = np.where(((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 3] == row[3])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4])))
                    meal_similar_vector = np.where( (meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 3] == row[3]) & (meal_matrix[:, 4] == row[4]) )

                    for with_food_row in meal_similar_vector[0]:
                        with_food = meal_matrix[with_food_row][2]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{lunch_or_dinner}\t{with_food}')

                elif i == 3:
                    # meal_similar_vector = np.where(((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 4] == row[4])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 4] == row[4])))
                    meal_similar_vector = np.where( (meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 4] == row[4]) )

                    for with_food_row in meal_similar_vector[0]:
                        with_food = meal_matrix[with_food_row][3]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{lunch_or_dinner}\t{with_food}')
                    
                else: # i = 4
                    # meal_similar_vector = np.where(((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 3] == row[3])) |
                    #                             ((meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3])) |
                    #                             ((meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3])))
                    meal_similar_vector = np.where( (meal_matrix[:, 0] == row[0]) & (meal_matrix[:, 1] == row[1]) & (meal_matrix[:, 2] == row[2]) & (meal_matrix[:, 3] == row[3]) )

                    for with_food_row in meal_similar_vector[0]:
                        with_food = meal_matrix[with_food_row][4]
                        if with_food != food and with_food != 'empty':
                            meal_similar_vectors.append(f'{food}\tsimilar_meal_level{lunch_or_dinner}\t{with_food}')

    return np.unique(meal_similar_vectors)

def get_meal_similar_vector(meal_similar_vector_breakfast, meal_similar_vector_lunch, meal_similar_vector_dinner):
    meal_similar_vector = np.unique(np.concatenate((meal_similar_vector_breakfast, meal_similar_vector_lunch, meal_similar_vector_dinner)))

    # rows_to_remove = []
    # for r in range(meal_similar_vector.shape[0]):
    #     row = meal_similar_vector[r]
    #     vector = row.split('\t')
    #     if vector[0] == 'empty' or vector[2] == 'empty':
    #         rows_to_remove.append(r)
    # meal_similar_vector = np.delete(meal_similar_vector, rows_to_remove, axis=0)

    return meal_similar_vector

def get_food_ingredient_matrix(food_vocab, ingredient_vocab, ingredient_DF):
    ingredient_names = np.array(ingredient_DF['name'])
    ingredient_to_idx = {ingredient: idx for idx, ingredient in enumerate(ingredient_vocab)}
    food_ingredient_matrix = np.zeros(( len(food_vocab), len(ingredient_vocab) ))

    for food_idx in range(len(food_vocab)):
        food = food_vocab[food_idx]
        match_ingredient_row = np.where(food == ingredient_names)[0]
        for row in match_ingredient_row:
            ingredient = np.array(ingredient_DF['ingredient'])[row]
            ingredient_idx = ingredient_to_idx[ingredient]
            food_ingredient_matrix[food_idx][ingredient_idx] = 1
    
    return food_ingredient_matrix

def get_ingredient_similar_vector(food_ingredient_matrix, theta, food_vocab):
    ingredient_similar_vector = []
    for row in range(food_ingredient_matrix.shape[0]):
        print(f"ingredient_similar_vector processing : {row / food_ingredient_matrix.shape[0] * 100:.2f}%") # process
        for row2 in range(food_ingredient_matrix.shape[0]):
            matches = np.sum( np.logical_and(food_ingredient_matrix[row], food_ingredient_matrix[row2]) )
            if matches >= theta:
                
                if row != row2:
                    ingredient_similar_vector.append(f'{food_vocab[row]}\tsimliar_ingredient_level\t{food_vocab[row2]}')

    return (ingredient_similar_vector)

def get_ingredient_similar_vector_ratio(food_ingredient_matrix, theta, food_vocab):
    ingredient_similar_vector = []
    num_foods = len(food_vocab)

    similarity_matrix = np.zeros((num_foods, num_foods))

    for row in range(food_ingredient_matrix.shape[0]):
        print(f"ingredient_similar_vector processing : {row / food_ingredient_matrix.shape[0] * 100:.2f}%") # process
        for row2 in range(food_ingredient_matrix.shape[0]):
            match_num = np.sum( np.logical_and(food_ingredient_matrix[row], food_ingredient_matrix[row2]) )
            union_num = np.sum(np.logical_or(food_ingredient_matrix[row], food_ingredient_matrix[row2]))

            food = food_vocab[row]
            with_food = food_vocab[row2]

            food_removed = re.sub(r'\(.*?\)', '', food).strip()
            with_food_removed = re.sub(r'\(.*?\)', '', with_food).strip()
            
            if union_num == 0:
                ratio = 0
            else:
                ratio = match_num / union_num
            
            if food != with_food:
                similarity_matrix[row, row2] = ratio


            if food_removed == with_food_removed and food != with_food:
                ingredient_similar_vector.append(f'{food}\tsimliar_ingredient_level\t{with_food}')

    similarity_matrix = min_max_normalize(similarity_matrix)

    for row in range(num_foods):
        for row2 in range(num_foods):
            food = food_vocab[row]
            with_food = food_vocab[row2]
            ratio = similarity_matrix[row, row2]
            if ratio >= theta and food != with_food:
                ingredient_similar_vector.append(f'{food}\tsimliar_ingredient_level\t{with_food}')



    return np.unique(ingredient_similar_vector)

def save_ingredient_name_similar_matrix(food_vocab, food_ingredient_matrix):
    num_foods = len(food_vocab)
    similarity_matrix = np.zeros((num_foods, num_foods))

    for row in range(num_foods):
        print(f"ingredient_name_similar_matrix processing : {row / food_ingredient_matrix.shape[0] * 100:.2f}%") # process
        for row2 in range(num_foods):
            match_num = np.sum( np.logical_and(food_ingredient_matrix[row], food_ingredient_matrix[row2]) )
            union_num = np.sum(np.logical_or(food_ingredient_matrix[row], food_ingredient_matrix[row2]))

            if union_num == 0:
                ratio = 0
            else:
                ratio = match_num / union_num

            if row != row2:
                similarity_matrix[row, row2] = ratio
    
    similarity_matrix = min_max_normalize(similarity_matrix)

    for row in range(num_foods):
        for row2 in range(num_foods):
            if row != row2:
                food = food_vocab[row]
                food_with = food_vocab[row2]

                food_removed = re.sub(r'\(.*?\)', '', food).strip()
                with_food_removed = re.sub(r'\(.*?\)', '', food_with).strip()
                if food_removed == with_food_removed and food != food_with:
                    similarity_matrix[row, row2] = 1.0

    similarity_matrix_tensor = torch.tensor(similarity_matrix, dtype=torch.float32)

    torch.save(similarity_matrix_tensor, 'C:/Users/UNIST/Desktop/ours/KG/data/v0.2/ingredient_name_similar_matrix.pt')
            

            
def min_max_normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:  
        return np.zeros_like(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def save_relation_vector(ingredient_similar_vector):
    # meal_ingredient_similar_vector = np.unique(np.concatenate((meal_similar_vector, ingredient_similar_vector)))
    meal_ingredient_similar_vector = np.unique(ingredient_similar_vector)

    relation_vector = []

    for vector in meal_ingredient_similar_vector:
        vector = vector.split('\t')
        food = vector[0]
        relation = vector[1]
        with_food = vector[2]

        # if not ((food in classes['class_0'] and with_food in classes['class_0']) or 
        #         (food in classes['class_1'] and with_food in classes['class_1']) or 
        #         (food in classes['class_2'] and with_food in classes['class_2']) or 
        #         (food in classes['class_3'] and with_food in classes['class_3']) or 
        #         (food in classes['class_4'] and with_food in classes['class_4'])):

        relation_vector.append(f'{food}\t{relation}\t{with_food}')

    relation_vector = np.unique(relation_vector)
    print(len(relation_vector))

    np.savetxt('C:/Users/UNIST/Desktop/ours/KG/data/relation_vector.txt', relation_vector, fmt='%s')

def get_name_similar_vector(food_vocab):
    name_similar_vector = []
    for food in food_vocab:
        for with_food in food_vocab:
            food_remove_brace = re.sub(r'\(.*?\)', '', food).strip()
            with_food_remove_brace = re.sub(r'\(.*?\)', '', with_food).strip()
            if food != with_food and food_remove_brace == with_food_remove_brace:
                name_similar_vector.append(f'{food}\tsimliar_name_level\t{with_food}')
    return name_similar_vector

def save_class_matirx(food_vocab):
    file_paths = {
        'class_0': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/밥.txt',
        'class_1': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/국.txt',
        'class_2': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/주찬.txt',
        'class_3': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/부찬.txt',
        'class_4': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/김치.txt',
    }

    classes = {}

    for key, file_path in file_paths.items():
        with open(file_path, 'r') as f:
            classes[key] = [line.strip() for line in f.readlines()]

    num_nodes = len(food_vocab)
    num_classes = len(classes)
    
    class_matrix = np.zeros((num_nodes, num_classes), dtype=np.float32)
    
    food_index = {food: idx for idx, food in enumerate(food_vocab)}

    for class_idx, (class_key, class_items) in enumerate(classes.items()):
        for item in class_items:
            if item in food_index:
                node_idx = food_index[item]
                class_matrix[node_idx, class_idx] = 1.0
    
    class_matrix =  torch.tensor(class_matrix)
    print(class_matrix.shape)
    torch.save(class_matrix, 'C:/Users/UNIST/Desktop/ours/KG/data/class_matrix.pt')


def get_class():
    file_paths = {
        'class_0': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/밥.txt',
        'class_1': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/국.txt',
        'class_2': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/주찬.txt',
        'class_3': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/부찬.txt',
        'class_4': 'C:/Users/UNIST/Desktop/ours/KG/data/class/processed/김치.txt',
    }

    classes = {}

    for key, file_path in file_paths.items():
        with open(file_path, 'r') as f:
            classes[key] = [line.strip() for line in f.readlines()]
    
    return classes


def save_allergy_matrix(food_vocab, allergy_DF, ingredient_DF):
    allergy_list = ['난류', '우유', '대두', '땅콩+견과류', '밀']
    allergy_matrix = np.zeros((len(food_vocab), len(allergy_list)))

    allergy_dict = {allergy: idx for idx, allergy in enumerate(allergy_DF.columns[1:])}
    allergy_to_matrix_idx = {allergy: idx for idx, allergy in enumerate(allergy_list)}

    for i, food in enumerate(food_vocab):
        ingredients = ingredient_DF[ingredient_DF['name'] == food]['ingredient'].values

        for ingredient in ingredients:
            allergy_row = allergy_DF[allergy_DF['Name'] == ingredient].iloc[0]
            for j, allergy in enumerate(allergy_list):
                if allergy_row[allergy] == 1:
                    allergy_matrix[i, j] = 1
    
    allergy_tensor = torch.tensor(allergy_matrix, dtype=torch.float32)
    torch.save(allergy_tensor, 'C:/Users/UNIST/Desktop/ours/KG/data/v0.2/allergy_matrix.pt')

    # egg_allergy_indices = (allergy_tensor[:, 4] == 1).nonzero(as_tuple=True)[0]

    # # Assuming you have the food_vocab list
    # egg_allergy_foods = [food_vocab[idx] for idx in egg_allergy_indices]

    # print(len(egg_allergy_foods))

def save_class_matrix_v2(food_with_voacb, nutri_DF, diets_DF):
    rice_class = np.unique(nutri_DF[ (nutri_DF['soup'] == 0) & (nutri_DF['snack'] == 0) & (nutri_DF['side_dish'] == 0) & (nutri_DF['kimchi'] == 0) & (nutri_DF['empty'] == 0)]['name']).tolist()
    soup_class = np.unique(nutri_DF[ (nutri_DF['soup'] == 1) ]['name']).tolist()
    side_dish_class = np.unique(nutri_DF[ (nutri_DF['side_dish'] == 1) ]['name']).tolist()
    kimchi_class = np.unique(nutri_DF[ (nutri_DF['kimchi'] == 1) ]['name']).tolist()

    diets_array = diets_DF.to_numpy()
    unique_foods = np.unique(diets_array)
    unique_foods = unique_foods[unique_foods != 'empty']
    snack_foods = np.unique(nutri_DF[nutri_DF['snack'] == 1]['name'])

    common_snack_foods = np.intersect1d(unique_foods, snack_foods)

    label_count_matrix = np.zeros((len(common_snack_foods), 4))

    for i in range(len(common_snack_foods)):
        food = common_snack_foods[i]
        matches = np.where(diets_DF == food)
        for _, col in zip(matches[0], matches[1]):
            if col in [0, 3, 8]: # 밥
                label_count_matrix[i][0] += 1
            elif col in [1, 4, 9]: # 국
                label_count_matrix[i][1] += 1
            elif col in [2, 5, 6, 10, 11]: # 주찬
                label_count_matrix[i][2] += 1
            else: # 김치
                label_count_matrix[i][3] += 1

    ambiguous_indices = []
    for i, row in enumerate(label_count_matrix):
        max_value = np.max(row)
        max_count = np.sum(row == max_value)
        if max_count >= 2:  
            ambiguous_indices.append(i)
    ambiguous_foods = common_snack_foods[ambiguous_indices]

    soup_class.append(ambiguous_foods[0]) # S단호박죽 [1. 1. 0. 0.]
    soup_class.append(ambiguous_foods[1]) # S단호박크림스프  [1. 1. 0. 0.]
    side_dish_class.append(ambiguous_foods[2]) # S떠먹는요구르트(100ml)  [0. 2. 2. 1.]
    side_dish_class.append(ambiguous_foods[3]) # S보리차(200ml)  [0. 1. 1. 0.]
    side_dish_class.append(ambiguous_foods[4]) # S오렌지주스(무가당)(200ml) [0. 1. 1. 0.]]

    class_label = np.argmax(label_count_matrix, axis=1)
    for i in range(len(common_snack_foods)):
        if i not in ambiguous_indices:
            food = common_snack_foods[i]
            class_idx = class_label[i]
            if class_idx == 0:
                rice_class.append(food)
            elif class_idx == 1:
                soup_class.append(food)
            elif class_idx == 2:
                side_dish_class.append(food)
            elif class_idx == 3:
                kimchi_class.append(food)

    class_matrix = np.zeros((len(food_with_voacb), 4))
    
    for i, food in enumerate(food_with_voacb):
        if food in rice_class:
            class_matrix[i] = [1, 0, 0, 0]
        elif food in soup_class:
            class_matrix[i] = [0, 1, 0, 0]
        elif food in side_dish_class:
            class_matrix[i] = [0, 0, 1, 0]
        elif food in kimchi_class:
            class_matrix[i] = [0, 0, 0, 1]

    class_matrix =  torch.tensor(class_matrix)
    torch.save(class_matrix, 'C:/Users/UNIST/Desktop/ours/KG/data/v0.2/class_matrix.pt')


if __name__ == "__main__":
    set_random_seed(42)
    diets_DF = pd.DataFrame(pd.read_csv('C:/Users/UNIST/Desktop/ours/KG/data/db/normal_diets_8234.csv'))
    ingredient_DF = pd.DataFrame(pd.read_csv('C:/Users/UNIST/Desktop/ours/KG/data/db/total_ingredient_data.csv'))
    allergy_DF = pd.DataFrame(pd.read_csv('C:/Users/UNIST/Desktop/ours/KG/data/db/allergy_db_updated.csv', encoding='euc-kr'))
    nutri_DF = pd.DataFrame(pd.read_csv('C:/Users/UNIST/Desktop/ours/KG/data/db/total_nutri_data.csv'))

    # food_vocab = get_food_vocab(ingredient_DF)
    ingredient_vocab = get_ingredient_vocab(ingredient_DF)
    food_with_voacb = get_food_with_vocab(diets_DF, nutri_DF)

    save_allergy_matrix(food_with_voacb, allergy_DF, ingredient_DF)

    # save_class_label(food_with_voacb, diets_DF)
    # class_label = get_class_label()
    # save_class_txt(class_label, food_with_voacb)
    # class_vector = get_class_vector(class_label, food_vocab)

    # save_class_matirx(food_with_voacb)
    # save_class_matrix_v2(food_with_voacb, nutri_DF, diets_DF)
    # process_snack_class(food_with_voacb, diets_DF)


    # breakfast_matrix, lunch_matrix, dinner_matrix = get_diet_matrix(diets_DF)
    # meal_similar_vector_breakfast = get_meal_similar_vector_breakfast(breakfast_matrix, '')
    # meal_similar_vector_lunch = get_meal_similar_vector_lunch_dinner(lunch_matrix, '')
    # meal_similar_vector_dinner = get_meal_similar_vector_lunch_dinner(dinner_matrix, '')
    # meal_similar_vector = get_meal_similar_vector(meal_similar_vector_breakfast, meal_similar_vector_lunch, meal_similar_vector_dinner)

    food_ingredient_matrix = get_food_ingredient_matrix(food_with_voacb, ingredient_vocab, ingredient_DF)
    ingredient_similar_vector = get_ingredient_similar_vector_ratio(food_ingredient_matrix, theta=0.5, food_vocab=food_with_voacb)


    save_relation_vector(ingredient_similar_vector)

    # save_ingredient_name_similar_matrix(food_with_voacb, food_ingredient_matrix)


    # print(len(meal_similar_vector))
    print(len(ingredient_similar_vector))

    # with_vector_index = np.unique(get_with_vector_index(food_with_vocab, diets_DF))
    # adj_matrix = get_adj_matrix(with_vector_index, food_vocab)
    # save_npz('C:/Users/UNIST/Desktop/RGCN/GenCAT/data/food/food_food_with_adj_matrix.npz', adj_matrix)


    
