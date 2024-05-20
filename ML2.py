import numpy as np
from sklearn.linear_model import LinearRegression


X_card = [
    [1, 0, 0, 0],  # {1}
    [0, 1, 0, 0],  # {2}
    [0, 0, 1, 0],  # {3}
    [0, 0, 0, 1],  # {4}
    [1, 0, 1, 0],  # {1,3}
    [1, 0, 0, 1],  # {1,4}
    [0, 1, 1, 0],  # {2,3}
    [0, 1, 0, 1],  # {2,4}
    [0, 0, 1, 1]   # {3,4}
]
y_card = [
    75,  # {1}
    77,  # {2}
    42,  # {3}
    45,  # {4}
    117, # {1,3}
    120, # {1,4}
    119, # {2,3}
    122, # {2,4}
    87   # {3,4}
]


model = LinearRegression()
model.fit(X_card, y_card)


predicted_utilities = model.predict(X_card)

X_ord = [
    [1, 0, 0, 1],  # {1,4}
    [0, 1, 0, 1]   # {2,4}
    
]
y_ord = [1, 0]  # {1,4} > {2,4}

X_ord2 = [
    [1, 0, 0, 1],  # {1,4}
    [0, 1, 0, 1],  # {2,4}
    [1, 0, 1, 0]   # {1,3}
]
y_ord2 =  [1, 0, 1]  # {1,4} > {2,4}, {1,3} > {1,4}

X_single_courses = [
    [1, 0, 1, 0],  # {1,3}
    [1, 0, 0, 1],  # {1,4}
    [0, 1, 1, 0],  # {2,3}
    [0, 1, 0, 1],  # {2,4}
    [0, 0, 1, 1]   # {3,4}
]
model.fit(X_ord2, y_ord2)
new_utilities = model.predict(X_single_courses)
print("new utilities:", new_utilities)


combinations = [
    
    '{1, 3}', '{1, 4}', '{2, 3}', '{2, 4}', '{3, 4}'
]
utility_dict = dict(zip(combinations, new_utilities))
print(utility_dict)

sorted_combinations = sorted(utility_dict.items(), key=lambda item: item[1], reverse=True)
top_two_combinations = sorted_combinations[:2]

print("utility")
for combination, utility in top_two_combinations:
    print(f" cors: {combination}  utility: {utility}")
