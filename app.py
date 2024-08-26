from flask import Flask, request, jsonify
import pandas as pd
from surprise import Dataset, Reader
from surprise import dump

# File paths (Update these paths according to your local setup or dataset location)
PROCESSED_EVENTS_PATH = 'data/processed/events_cleaned.csv'
MODEL_PATH = 'models/recommendation_model.svd'

app = Flask(__name__)

# Load preprocessed data
events = pd.read_csv(PROCESSED_EVENTS_PATH)

# Load the trained model
_, loaded_model = dump.load(MODEL_PATH)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(events[['user_id', 'item_id', 'rating']], reader)
trainset = data.build_full_trainset()

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if user_id is None:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        # Convert user_id to int if needed
        user_id = int(user_id)

        # Get all items
        all_items = trainset.all_items()

        # Generate predictions for all items for the given user
        predictions = [loaded_model.predict(user_id, trainset.to_raw_iid(item_id)) for item_id in all_items]

        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top 10 recommendations
        top_predictions = predictions[:10]

        # Convert to a list of item ids
        recommendations = [pred.iid for pred in top_predictions]

        return jsonify({'user_id': user_id, 'recommendations': recommendations})
    except ValueError:
        return jsonify({'error': 'Invalid User ID format'}), 400

@app.route('/')
def index():
    return "Recommendation Engine is running."

if __name__ == '__main__':
    app.run(debug=True)
