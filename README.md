# LSH using Jaccard

This module implements Locality Sensitive Hashing (LSH) using Jaccard similarity to identify similar businesses from Yelp review data using PySpark.

## Overview

- **Data Preprocessing**: Loads and processes the Yelp dataset, creating a characteristic matrix where each business is represented by a vector of user interactions.
- **LSH & Signature Matrix**: Applies LSH to reduce the dimensionality of the data and generates a signature matrix for efficient similarity estimation.
- **Candidate Pairs & Jaccard Similarity**: Generates candidate business pairs and calculates their Jaccard similarity. Pairs with a similarity ≥ 0.5 are retained.

Check out the latest improved version of this Project here - https://github.com/revatip2/recommender-system-yelp-II.git 
