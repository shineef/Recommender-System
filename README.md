# For this Book Recommender System

## Overview

This project implements a book recommendation system using collaborative filtering approaches: Matrix Factorization (MF) with TensorFlow/Keras and Neural Collaborative Filtering (NCF) with PyTorch. The system includes features for adding and deleting user information, rating books, and generating book recommendations.

## System Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

## Installation

1. Install Python 3.x from [Python.org](https://www.python.org/downloads/).
2. Install the required libraries using pip:
   
   ```bash
   pip install numpy pandas scikit-learn matplotlib tensorflow keras torch tqdm
   ```

## File Structure

- `MF.py`: Implements the Matrix Factorization model using TensorFlow and Keras.
- `NCF_torch.py`: Implements the Neural Collaborative Filtering model using PyTorch.
- `main.py`: Main script to run the recommendation system, including item-based CF, loading model and user interaction for book recommendations.

## How to Run

1. Clone the repository:
   
   ```bash
   git clone https://github.com/shineef/Recommender-System.git
   ```
2. Navigate to the project directory.
3. Run the `main.py` script:
   
   ```bash
   python main.py
   ```

## Usage

- The script will prompt for the user's ID and the desired action (rate, search, delete, modify, or exit).
- For rating, enter the user's location, age, book ISBN, and rating.
- For book recommendations, the system will provide a list of suggested books based on the chosen model (MF or NCF) and user's past ratings.
- User information can be added or deleted as needed.

## Note

- GPU support for TensorFlow and Pytorch can be checked and utilized if available, which significantly speeds up training.
- Ensure that the datasets (`Books.csv`, `Ratings.csv`, `Users.csv`) are correctly placed in the specified paths.

## Data

Dataset from: [Book Recommendation Dataset | Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

## Additional Information

- The project is designed to be extensible, allowing for modifications and integrations of different recommendation algorithms.
- For detailed understanding of the code and algorithms, refer to the comments in the source files.
