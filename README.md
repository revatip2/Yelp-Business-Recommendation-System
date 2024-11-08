## LSH using Jaccard

This module implements Locality Sensitive Hashing (LSH) using Jaccard similarity to identify similar businesses from Yelp review data using PySpark.

### Overview

- **Data Preprocessing**: Loads and processes the Yelp dataset, creating a characteristic matrix where each business is represented by a vector of user interactions.
- **LSH & Signature Matrix**: Applies LSH to reduce the dimensionality of the data and generates a signature matrix for efficient similarity estimation.
- **Candidate Pairs & Jaccard Similarity**: Generates candidate business pairs and calculates their Jaccard similarity. Pairs with a similarity ≥ 0.5 are retained.

## Pearson Similarity-based Collaborative Filtering

This module implements an Item-Item Collaborative Filtering (CF) recommender system using Pearson similarity to predict ratings for businesses based on user interactions.

### Overview

- **Data Preprocessing**: Loads Yelp training and validation data, and builds mappings of businesses to users and users to businesses.
- **Rating Calculations**: Computes the average ratings for businesses and prepares user ratings data.
- **Pearson Similarity**: Calculates the Pearson correlation between businesses based on common users' ratings.
- **Prediction**: For each user-business pair in the validation set, predicts the rating using a weighted sum of similar businesses' ratings.
- **Output**: Writes the predicted ratings for each user-business pair to the output file.

## XGBoost Regressor-based Rating Prediction

This module uses an XGBoost Regressor model to predict Yelp ratings based on user and business features.

### Overview

- **Data Preprocessing**: Loads and processes Yelp training and validation data, along with user and business metadata (from JSON files), extracting relevant features.
- **Feature Engineering**: Constructs user and business-related features such as review count, average stars, and business review counts, alongside user ratings.
- **Model Training**: Uses XGBoost Regressor to train a model based on the extracted features.
- **Prediction**: Predicts ratings for the validation dataset.
