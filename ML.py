import numpy as np
from sklearn.linear_model import LinearRegression

# Function to prepare data for model training
def prepare_data(Di_card, Di_ord):
    X = []
    y = []

    for score, courses in Di_card:
        feature_vector = np.zeros(len(Di_card))
        for course in courses:
            feature_vector[course-1] = 1
        X.append(feature_vector)
        y.append(score)

    return np.array(X), np.array(y)

# Function to train the model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict scores for new combinations
def predict_scores(model, Di_card):
    X_new = []
    for _, courses in Di_card:
        feature_vector = np.zeros(len(Di_card))
        for course in courses:
            feature_vector[course-1] = 1
        X_new.append(feature_vector)
    return model.predict(np.array(X_new))

# First dataset
Di_card_1 = [
    (75, {1}),
    (77, {2}),
    (42, {3}),
    (45, {4}),
    (117, {1,3}),
    (120, {1,4}),
    (119, {2,3}),
    (122, {2,4}),
    (87, {3,4})
]
Di_ord_1 = [{1,4}>{2,4}]

# Prepare the first dataset
X1, y1 = prepare_data(Di_card_1, Di_ord_1)

# Train the model on the first dataset
model1 = train_model(X1, y1)

# Predict the best combinations for the first dataset
predicted_scores_1 = predict_scores(model1, Di_card_1)
best_combinations_1 = sorted(zip(predicted_scores_1, Di_card_1), key=lambda x: -x[0])[:2]

# Update course ratings for the first dataset
updated_Di_card_1 = [(int(score), courses) for score, (_, courses) in zip(predicted_scores_1, Di_card_1)]

# Print results for the first dataset
print("Top two combinations for the first dataset:", best_combinations_1)
print("Updated course ratings for the first dataset:", updated_Di_card_1)

# Second dataset
Di_card_2 = [
    (80, {1}),
    (72, {2}),
    (42, {3}),
    (45, {4}),
    (122, {1,3}),
    (125, {1,4}),
    (114, {2,3}),
    (117, {2,4}),
    (87, {3,4})
]
Di_ord_2 = [{1,4}>{2,4}, {1,3}>{1,4}, {1,3}>{2,4}]

# Prepare the second dataset
X2, y2 = prepare_data(Di_card_2, Di_ord_2)

# Train the model on the second dataset
model2 = train_model(X2, y2)

# Predict the best combinations for the second dataset
predicted_scores_2 = predict_scores(model2, Di_card_2)
best_combinations_2 = sorted(zip(predicted_scores_2, Di_card_2), key=lambda x: -x[0])[:2]

# Update course ratings for the second dataset
updated_Di_card_2 = [(int(score), courses) for score, (_, courses) in zip(predicted_scores_2, Di_card_2)]

# Print results for the second dataset
print("Top two combinations for the second dataset:", best_combinations_2)
print("Updated course ratings for the second dataset:", updated_Di_card_2)
