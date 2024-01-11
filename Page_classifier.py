import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances, precision_recall_fscore_support, accuracy_score


def pageRank(graph, input_images, beta=0.85, epsilon=0.000001):
    nodes = len(graph)

    # Initialize teleportation matrix and Page Rank Scores with zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)

    # Updating teleportation and Page Rank Score matrices with 1/num_of_input images for the input images.
    for image_idx in input_images:
        teleportation_matrix[image_idx] = 1 / len(input_images)
        pageRankScores[image_idx] = 1 / len(input_images)

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(graph, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break

    # Normalizing & Returning Page Rank Scores
    return pageRankScores / np.sum(pageRankScores)





df_even = pd.read_csv("RESNET_even.csv", header=None)
source_vec_even = df_even.loc[:,2:]

# source_vec_even=source_vec_even.T
image_id_lst_even = df_even[0].tolist()
label_id_lst_even = df_even[1].tolist()
unique_label_id_lst = list(set(df_even[1].tolist()))
num_images_even, num_labels_even = len(df_even[0].tolist()), np.max(df_even[1].tolist())


df_odd = pd.read_csv("RESNET_odd.csv", header=None)
source_vec_odd = df_odd.loc[:,2:]

# source_vec_odd=source_vec_odd.T
image_id_lst_odd = df_odd[0].tolist()
label_id_lst_odd = df_odd[1].tolist()

num_images_odd, num_labels_odd = len(df_odd[0].tolist()), np.max(df_odd[1].tolist())
num_features = len(source_vec_even)

...


distances = euclidean_distances(source_vec_even)
graph = 1 / (1 + distances)  # add 1 to avoid division by zero
graph = graph / np.sum(graph, axis=0)  # normalize to get transition probabilities


# Compute Personalized PageRank for each unique label
unique_labels = np.unique(label_id_lst_even)
output = []
for label in unique_labels:
  print(f'Calculating Personalized PageRank for label: {label}')
  input_images = np.where(label_id_lst_even == label)[0]
  output.append([label, pageRank(graph, input_images, beta=0.85)])

# Assigning label to each image based on the highest PageRank score
predicted_labels = []
for i in range(len(label_id_lst_odd)):
  compare = [[output[j][0], output[j][1][i]] for j in range(len(output))]
  predicted_labels.append(max(compare, key=lambda x: x[1])[0])

# Compute precision, recall, F1-score, and accuracy
precision, recall, f1, _ = precision_recall_fscore_support(label_id_lst_odd, predicted_labels)
accuracy = accuracy_score(label_id_lst_odd, predicted_labels)

# Print precision, recall, F1-score, and accuracy for each label and overall
for i, label in enumerate(unique_labels):
  print(f'Label: {label}')
  print(f'Precision: {precision[i]}')
  print(f'Recall: {recall[i]}')
  print(f'F1-score: {f1[i]}\n')
  print(f'Overall accuracy: {accuracy}')