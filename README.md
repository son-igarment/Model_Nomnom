# Food Recommendation System

## Developed by Phạm Lê Ngọc Sơn

This project implements a food recommendation system using semantic search and embedding models to find similar recipes based on name, ingredients, tags, and nutritional information.

## Project Structure

```
.
├── scripts/                   # Main code directory
│   ├── apis/                  # API implementations
│   ├── data/                  # Data storage directory
│   ├── model/                 # Model implementations
│   ├── app.py                 # Flask application for serving the model
│   ├── constant.py            # Constants used throughout the project
│   ├── data_reader.ipynb      # Notebook for data reading and exploration
│   ├── food-recommendation-systems.ipynb  # Core recommendation system notebook
│   ├── main.py                # Main entry point for the application
│   ├── offical_preprocess.ipynb  # Data preprocessing notebook
│   ├── similar_foods.py       # Implementation for finding similar foods
│   └── subtag_set.txt         # List of food tags used in the system
├── which_to_nomnom.ipynb      # Main notebook for running the entire pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore file
```

## Installation

The project requires approximately 1.5GB of disk space for transformer models and dependencies.

```bash
pip install -r requirements.txt
```

Download the necessary data files into the appropriate locations:

```bash
cd scripts
gdown 1i3eT_VF6yA_G4GvlAsehwSeIQmzXzn40 -O data/RAW_recipes.csv
gdown 1-7p4bHR2IAWAaZHbaRS-IUYWb1TMsvWS -O data/embedded_names.pkl
gdown 1-6Rvib4upv9VHEl1nwB2-D1SczbTieD2 -O data/embedded_ingredients.pkl
gdown 1yZQi3gWc90xGwXDvwvuTTMGbzzzkl1Gc -O data/embedded_tags.pkl
gdown 1nL4rOEbZiEEVqM1EdDL7WpifMcparEax -O data/scaler.pkl
gdown 1-9ee6DlGTn5RhmONWQdqGzCdtiGP4POr -O data/nutrition_scaled.pkl
```

## Quick Start

The easiest way to use the project is to run all cells in `which_to_nomnom.ipynb`, which sets up the environment, loads the data, and initializes the model.

## Running the Application

To run the application server:

```bash
cd scripts
python app.py
```

This will start a Flask server that provides an API for food recommendations.

## Model Usage

The recommendation model uses embeddings created with SentenceTransformer to find similar foods based on various criteria:

```python
# Example usage
model.search(
    name='salmon',
    ingredients='salmon, wasabi',
    tags=['japanese', '15-minutes-or-less'],
    nutrition=[600, 80, 10, 50, 150, 30, 50] 
    # NUTRITIONS = ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
)
```

### Parameters

- `name`: String containing the recipe name to search for
- `ingredients`: String containing comma-separated ingredients
- `tags`: List of tags (cuisine type, meal type, preparation time, etc.)
- `nutrition`: List of nutritional values in the order: calories, fat, sugar, sodium, protein, saturated_fat, carbohydrates
- `k`: Maximum number of results to consider (higher values give better results but are slower)

### Return Value

The method returns a list of tuples, each containing food information and a similarity score. Higher scores indicate greater similarity.

### Retrieving Recipe IDs

```python
# Get the IDs of the recommended foods
ids = [similar_food[0]['id'] for similar_food in similar_foods]
```

## Advanced Usage

The `BetterSearchModel` class provides additional filtering options:

```python
# Must have at least one of the specified tags
model.search(
    name='chicken',
    tags=['dinner', 'quick'],
    must_have_tags=True
)

# Must have all specified tags
model.search(
    name='dessert',
    tags=['chocolate', 'easy'],
    must_have_tags=True,
    must_have_all_tags=True
)
```

## Technical Details

The model uses:
- SentenceTransformer for embeddings (sentence-transformers/all-mpnet-base-v2)
- FAISS for efficient similarity search
- StandardScaler for normalizing nutritional values
- Flask for API serving

## Dataset

The system uses the Food.com recipe dataset, which includes recipe names, ingredients, preparation steps, nutritional information, and user-assigned tags.

## Contact

For more information, please contact Phạm Lê Ngọc Sơn.