import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, normalize

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False)
label_encoder = LabelEncoder()
example_array = np.random.rand(5, 2) * 90 + 10
animals = ['cat', 'dog', 'fish', 'dog', 'cat']

print("Original array:")
print(example_array)

std_scaled_array = std_scaler.fit_transform(example_array)
print("\nStandard Scaled array:")
print(std_scaled_array)

minmax_scaled_array = minmax_scaler.fit_transform(example_array)
print("\nMin-Max Scaled array:")
print(minmax_scaled_array)

normalized_array = normalize(example_array)
print("\nNormalized array:")
print(normalized_array)

one_hot_array = one_hot_encoder.fit_transform(np.array(animals).reshape(-1, 1))
print("\nOne-Hot Encoded array:")
print(one_hot_array)

label_encoded_array = label_encoder.fit_transform(animals)
print("\nLabel Encoded array:")
print(label_encoded_array)
