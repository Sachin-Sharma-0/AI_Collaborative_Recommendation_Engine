import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import dump

# File paths
PROCESSED_EVENTS_PATH = 'data/processed/events_cleaned.csv'
MODEL_PATH = 'models/recommendation_model.svd'

def load_preprocessed_data():
    events = pd.read_csv(PROCESSED_EVENTS_PATH)
    return events

def prepare_data_for_surprise(events):
    # Print column names to debug
    print("Columns in events:", events.columns)
    
    # Convert to Surprise dataset format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(events[['user_id', 'item_id', 'rating']], reader)
    return data

def train_model(data):
    # Split the data into training and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Initialize the SVD algorithm
    algo = SVD()

    # Train the algorithm on the trainset
    algo.fit(trainset)

    return algo

def save_model(algo):
    # Save the trained model
    dump.dump(MODEL_PATH, algo=algo)

def main():
    # Load preprocessed data
    events = load_preprocessed_data()

    # Prepare data for Surprise
    data = prepare_data_for_surprise(events)

    # Train the model
    algo = train_model(data)

    # Save the trained model
    save_model(algo)
    print("Model training complete. Model saved.")

if __name__ == '__main__':
    main()
