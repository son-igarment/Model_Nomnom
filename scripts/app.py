# Food Recommendation System
# Developed by Phạm Lê Ngọc Sơn

import pickle
import time
from flask import Flask

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
import os
import faiss
import ast
import time
from sklearn.preprocessing import StandardScaler

class SearchModel():
    def __init__(self, nutritions_scaler=None):
        self.data = {}
        self.weights = [1.0, 1.0, 1.0, 1.0]
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.index_for_name = None # Index is the model by faiss
        self.index_for_tags = None
        self.index_for_ingredients = None
        self.index_for_nutrition = None
        self.scaler = nutritions_scaler

    def embed(self, sentences):
        '''
            sentences: list of strings
            return: list of embeddings
        '''
        return self.embedding_model.encode(sentences)

    def fit_name(self, embeddings):
        self.index_for_name = faiss.IndexFlatIP(embeddings.shape[1])  # IndexFlatIP for inner product (cosine similarity)
        self.index_for_name.add(embeddings)

    def fit_tag(self, embeddings):
        self.index_for_tags = faiss.IndexFlatIP(embeddings.shape[1])  # IndexFlatIP for inner product (cosine similarity)
        self.index_for_tags.add(embeddings)

    def fit_ingredients(self, embeddings):
        self.index_for_ingredients = faiss.IndexFlatIP(embeddings.shape[1])  # IndexFlatIP for inner product (cosine similarity)
        self.index_for_ingredients.add(embeddings)

    def fit_nutrition(self, nutrition):
        nutrition = np.ascontiguousarray(nutrition, dtype=np.float32)
        self.index_for_nutrition = faiss.IndexFlatL2(7)  # IndexFlatL1 for L1 distance (Manhattan)
        self.index_for_nutrition.add(nutrition)

    def fit(self, embedded_names, embedded_tags, embedded_ingredients, nutritions_scaled):
        self.fit_name(embedded_names)
        self.fit_tag(embedded_tags)
        self.fit_ingredients(embedded_ingredients)
        self.fit_nutrition(nutritions_scaled)

    def load(self, data):
        '''
            data: dataframe
        '''
        print('[+] Loading data into model...')
        self.data = data

    def search(self, name: str=None, tags: str=None, ingredients: str=None, nutrition: list[float]=None, k=1000):
        '''
            return: list of similar
        '''
        print('[+] Retrieving similar foods...')
        final_res = {}
        if name:
            embedded_name = self.embedding_model.encode(name)
            name_scores, name_indices = self.index_for_name.search(embedded_name.reshape(1, -1), k=k)
            for i, idx in enumerate(name_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['name'] = name_scores[0][i] * self.weights[0]

        if tags:
            embedded_tags = self.embedding_model.encode(tags)
            tags_scores, tags_indices = self.index_for_tags.search(embedded_tags.reshape(1, -1), k=k)
            for i, idx in enumerate(tags_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['tags'] = tags_scores[0][i] * self.weights[1]

        if ingredients:
            embedded_ingredients = self.embedding_model.encode(ingredients)
            ingredients_scores, ingredients_indices = self.index_for_ingredients.search(embedded_ingredients.reshape(1, -1), k=k)
            for i, idx in enumerate(ingredients_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['ingredients'] = ingredients_scores[0][i] * self.weights[2]

        if nutrition:
            scaled_nutrition = np.array(nutrition).reshape(1, -1)
            scaled_nutrition = self.scaler.transform(scaled_nutrition)
            nutrition_scores, nutrition_indices = self.index_for_nutrition.search(scaled_nutrition.reshape(1, -1), k=k)
            max_nutrition_scores = max(nutrition_scores[0])
            normalized_nutrition_scores = [1 - score/max_nutrition_scores for score in nutrition_scores[0]]
            for i, idx in enumerate(nutrition_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['nutrition'] = normalized_nutrition_scores[i] * self.weights[3]

        final_list = [
            (idx, sum([score for crit, score in scores.items()])) for idx, scores in final_res.items()
        ]
        print(len(final_list))

        return [(self.data.iloc[idx], score) for idx, score in sorted(final_list, key=lambda x: x[1], reverse=True)]

class BetterSearchModel(SearchModel):
    def search(self, *, name: str=None, tags: list[str]=None, ingredients: str=None, nutrition: list[float]=None, k=1000, must_have_tags=False, must_have_all_tags=False):
        '''
            return: list of similar
        '''
        print('[+] Retrieving similar foods...')
        final_res = {}
        if name:
            embedded_name = self.embedding_model.encode(name)
            name_scores, name_indices = self.index_for_name.search(embedded_name.reshape(1, -1), k=k)
            for i, idx in enumerate(name_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['name'] = name_scores[0][i] * self.weights[0]

        if tags:
            concat_tags = ' '.join(tags)

            embedded_tags = self.embedding_model.encode(concat_tags)
            tags_scores, tags_indices = self.index_for_tags.search(embedded_tags.reshape(1, -1), k=k)
            for i, idx in enumerate(tags_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['tags'] = tags_scores[0][i] * self.weights[1]

        if ingredients:
            embedded_ingredients = self.embedding_model.encode(ingredients)
            ingredients_scores, ingredients_indices = self.index_for_ingredients.search(embedded_ingredients.reshape(1, -1), k=k)
            for i, idx in enumerate(ingredients_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['ingredients'] = ingredients_scores[0][i] * self.weights[2]

        if nutrition:
            scaled_nutrition = np.array(nutrition).reshape(1, -1)
            scaled_nutrition = self.scaler.transform(scaled_nutrition)
            nutrition_scores, nutrition_indices = self.index_for_nutrition.search(scaled_nutrition.reshape(1, -1), k=k)
            max_nutrition_scores = max(nutrition_scores[0])
            normalized_nutrition_scores = [1 - score/max_nutrition_scores for score in nutrition_scores[0]]
            for i, idx in enumerate(nutrition_indices[0]):
                if idx not in final_res:
                    final_res[idx] = {}
                final_res[idx]['nutrition'] = normalized_nutrition_scores[i] * self.weights[3]

        food_list = [
            (idx, sum([score for crit, score in scores.items()])) for idx, scores in final_res.items()
        ]
        filtered_food_list = []
        print(len(food_list))

        if tags and must_have_tags:
            df = self.data[self.data.index.isin([idx for idx, score in food_list])]
            print(tags, len(df))
            if must_have_all_tags:
                df = df[df['tags'].apply(lambda food_tags: all(tag in food_tags for tag in tags))]
            else:
                df = df[df['tags'].apply(lambda food_tags: any(tag in food_tags for tag in tags))]
            filtered_food_list = [(idx, score) for idx, score in food_list if idx in df['id']]
            print(len(filtered_food_list))

        final_food_list = filtered_food_list if filtered_food_list else food_list

        return [(self.data.iloc[idx], score) for idx, score in sorted(final_food_list, key=lambda x: x[1], reverse=True)]

EMBEDDED_NAME_PATH = 'data/embedded_names.pkl'
EMBEDDED_TAGS_PATH = 'data/embedded_tags.pkl'
EMBEDDED_INGR_PATH = 'data/embedded_ingredients.pkl'
SCALER_PATH = 'data/scaler.pkl'
NUTRITION_SCALED_PATH = 'data/nutrition_scaled.pkl'

with open(EMBEDDED_NAME_PATH, 'rb') as f:
	embedded_names = pickle.load(f)
with open(EMBEDDED_TAGS_PATH, 'rb') as f:
	embedded_tags = pickle.load(f)
with open(EMBEDDED_INGR_PATH, 'rb') as f:
	embedded_ingredients = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
	scaler = pickle.load(f)
with open(NUTRITION_SCALED_PATH, 'rb') as f:
	nutritions_scaled = pickle.load(f)  

df = pd.read_csv('data/RAW_recipes.csv')
df['name'] = df['name']
df['tags'] = df['tags'].apply(ast.literal_eval)
df['ingredients'] = df['ingredients'].apply(ast.literal_eval).apply(lambda x: ', '.join(x))
df['nutrition'] = df['nutrition'].apply(ast.literal_eval)
NUTRITIONS = ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
# Unpack the "Nutrition" column into separate columns
df[NUTRITIONS] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

bettermodel = BetterSearchModel(scaler)
bettermodel.load(df)
bettermodel.fit(embedded_names, embedded_tags, embedded_ingredients, nutritions_scaled)

app = Flask(__name__)

@app.route("/")
def test():
    similar_foods = bettermodel.search(
        name='salmon',
        ingredients='salmon, wasabi',
        tags='japanese, 15-minutes-or-less',
        nutrition=[600, 80, 10, 50, 150, 30, 50], # NUTRITIONS = ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    )[:10] # Take 10 most similar (can take up to k)
    
    names = [similar_food[0]['name'] for similar_food in similar_foods]
    ids = [similar_food[0]['id'] for similar_food in similar_foods]
    print(names)
    print(ids)
    
    return names

if __name__ == '__main__':
    app.run()