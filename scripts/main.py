# Food Recommendation System
# Developed by Phạm Lê Ngọc Sơn

from model.model import EmbeddingModel
from data.dataReader import DataReader
import pickle
import os
# load DATA_PATH and MODEL_PATH
DATA_PATH = 'data/RAW_recipes.csv'
MODEL_PATH = 'scripts/model/embeddings.pkl'

# data reader
data_reader = DataReader(DATA_PATH)
data = data_reader.load_data_embed_name() #only embedding the name of food
itemList = data_reader.dataframe_creator()
# create model

embeddings = None

model = EmbeddingModel()

# load data to model
model.read_data(data)

if not os.path.exists(MODEL_PATH):
	# embedding
	embeddings = model.embedding(data)

	# pkl
	with open('../embeddings.pkl', 'wb') as f:
		pickle.dump(embeddings, f)

else:
	with open(MODEL_PATH, 'rb') as f:
		embeddings = pickle.load(f)
		model.embeddings = embeddings

# print(embeddings)
# Give a input sentence, retrieve which embeddings are most similar to the input sentence

# test find food information from input query
# input_sentence = 'fried chicken'

# retrieve similar embeddings
# similar_embeddings = model.similar_embeddings(input_sentence)

# print(similar_embeddings)

# print(similar_embeddings[0][1])

# retrieve information for similar_embeddings[0] (top 1)
# each embedding is a tuple (similarity_score, food_name), need to retrieve with same food name

# for i in range(len(itemList)):
#     if similar_embeddings[0][1] == itemList[i][0]:
#         print(itemList[i])
#         break

# test find foods based on tags
# expected = []
# tagsList = ['preparation', 'number-of-servings']
# for item in itemList:
#     if all(tag in item[2] for tag in tagsList):
#         expected.append(item)

# print(expected[:2])
		
# combinations of tags and embeddings
# input_sentence = 'fried chicken'
# tags = ['preparation', 'number-of-servings']
# expected = []
# tagsList = ['preparation', 'number-of-servings']
# for item in itemList:
#     if all(tag in item[2] for tag in tagsList):
#         expected.append(item)

# # get name list
# name_from_item = []
# for item in expected:
# 	name_from_item.append(item[0])

# temporary_model = EmbeddingModel()
# temporary_model.read_data(name_from_item)
# temporary_embedding = temporary_model.embedding(name_from_item)
# similar_embeddings = temporary_model.similar_embeddings(input_sentence)

# for i in range(len(expected)):
#     if similar_embeddings[0][1] == expected[i][0]:
#         print(expected[i])
#         break
