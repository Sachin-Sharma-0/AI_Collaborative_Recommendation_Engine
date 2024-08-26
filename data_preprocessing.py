import pandas as pd

# File paths (Update these paths according to your local setup or dataset location)
CATEGORY_TREE_PATH = 'data/retailrocket/category_tree.csv'
EVENTS_PATH = 'data/retailrocket/events.csv'
ITEM_PROPERTIES_PART1_PATH = 'data/retailrocket/item_properties_part1.csv'
ITEM_PROPERTIES_PART2_PATH = 'data/retailrocket/item_properties_part2.csv'

def load_data():
    category_tree = pd.read_csv(CATEGORY_TREE_PATH)
    events = pd.read_csv(EVENTS_PATH)
    item_properties_part1 = pd.read_csv(ITEM_PROPERTIES_PART1_PATH)
    item_properties_part2 = pd.read_csv(ITEM_PROPERTIES_PART2_PATH)
    
    # Combine item properties parts
    item_properties = pd.concat([item_properties_part1, item_properties_part2], ignore_index=True)
    
    return events, item_properties, category_tree

def preprocess_events(events):
    # Convert timestamps to datetime if needed
    print("Columns in events before processing:", events.columns)
    if 'timestamp' in events.columns:
        # Filter out invalid timestamps
        valid_timestamp_mask = events['timestamp'] > 0
        events = events[valid_timestamp_mask]
        events['timestamp'] = pd.to_datetime(events['timestamp'], unit='s', errors='coerce')

    # Ensure the correct column names
    if 'visitorid' in events.columns:
        events.rename(columns={'visitorid': 'user_id'}, inplace=True)
    if 'itemid' in events.columns:
        events.rename(columns={'itemid': 'item_id'}, inplace=True)
    # Handle missing values
    events.fillna({'category_id': -1, 'item_id': -1}, inplace=True)

    # Remove duplicates
    events.drop_duplicates(inplace=True)

    # Create a rating column based on event types
    if 'event' in events.columns:
        events['rating'] = events['event'].apply(lambda x: 5 if x == 'transaction' else (3 if x == 'addtocart' else 1))
    else:
        events['rating'] = 1  # Default rating if event column doesn't exist

    print("Columns in events after processing:", events.columns)
    return events


def preprocess_item_properties(item_properties):
    # Handle missing values
    item_properties.fillna({'category_id': -1}, inplace=True)

    # Remove duplicates
    item_properties.drop_duplicates(inplace=True)

    return item_properties

def preprocess_category_tree(category_tree):
    # Handle missing values
    category_tree.fillna({'parent_category_id': -1}, inplace=True)

    # Remove duplicates
    category_tree.drop_duplicates(inplace=True)

    return category_tree

def save_preprocessed_data(events, item_properties, category_tree):
    events.to_csv('data/processed/events_cleaned.csv', index=False)
    item_properties.to_csv('data/processed/item_properties_cleaned.csv', index=False)
    category_tree.to_csv('data/processed/category_tree_cleaned.csv', index=False)

def main():
    # Load raw data
    events, item_properties, category_tree = load_data()

    # Preprocess data
    events = preprocess_events(events)
    item_properties = preprocess_item_properties(item_properties)
    category_tree = preprocess_category_tree(category_tree)

    # Save preprocessed data
    save_preprocessed_data(events, item_properties, category_tree)
    print("Data preprocessing complete. Cleaned files saved.")

if __name__ == '__main__':
    main()
