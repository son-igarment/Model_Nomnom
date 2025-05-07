from model.model import SearchModel
import pickle
import time

if __name__ == '__main__':
    with open('model/model0604.pkl', 'rb') as f:
        model: SearchModel = pickle.load(f)
    
    similar_foods = model.search(
        name='salmon',
        ingredients='salmon, wasabi',
        tags='japanese, 15-minutes-or-less',
        nutrition=[600, 80, 10, 50, 150, 30, 50], # NUTRITIONS = ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    )[:10] # Take 10 most similar (can take up to k)

    names = [similar_food[0]['name'] for similar_food in similar_foods]
    ids = [similar_food[0]['id'] for similar_food in similar_foods]
    print(names)
    print(ids)
