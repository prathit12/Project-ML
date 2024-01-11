import numpy as np
import math
import torch
import pymongo
from torchvision import datasets
import os
import math
import operator

def Euclideandist(x,xi, length):
    d = 0.0
    for i in range(length):
        d += pow(float(x[i])- float(xi[i]),2)
    return math.sqrt(d)


class MNNClassifier:
    def getNeighbors(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = Euclideandist(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # After sorting the neighbours based on their respective classes, max voting to give the final class of the test instance
    import operator
    def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)  # Sorting it based on votes
        return sortedVotes[0][0]  # Please note we need the class for the top voted class, hence [0][0]#



def perform_pca(data_list, num_components):
    data = np.array(data_list)
    num_rows, num_features = data.shape[0], data.shape[1]
    data = data.reshape(num_rows, -1)

    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    data_std = (data - mean) / std_dev

    cov_matrix = np.cov(data_std, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    eigenvalue_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_idx]
    eigenvectors = eigenvectors[:, eigenvalue_idx]

    selected_eigenvectors = eigenvectors[:, :num_components]
    new_data = np.dot(data_std, selected_eigenvectors)

    return new_data


def main():
    # Connect to MongoDB
    database = pymongo.MongoClient("mongodb://localhost:27017/")
    db = database["feature_descriptor_db_2"]
    collection = db["color_moments_2"]

    even_images = list(collection.find({"_id": {"$mod": [2, 0]}}))

    even_image_fd_list = [np.array(doc["fd"]) for doc in even_images]

    odd_images = list(collection.find({"_id": {"$mod": [2, 1]}}))
    odd_image_fd_list = [np.array(doc["fd"]) for doc in odd_images]

    odd_labels = [np.array(doc["label"]) for doc in odd_images]

    even_labels = [np.array(doc["label"]) for doc in even_images]

    even_image_fd_list = perform_pca(even_image_fd_list, 1)
    odd_image_fd_list = perform_pca(odd_image_fd_list, 1)

    data_dir = r'C:\Users\91816\Downloads\CSE-515-Phase1\CSE-515-Phase1\Code\caltech101\101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir)


    predictions = []
    m = int(input("Enter the value of 'm' for k-NN: "))
    for x in range(len(odd_image_fd_list)):
        neighbors = MNNClassifier.getNeighbors(even_image_fd_list, odd_labels, m)
        result = MNNClassifier.getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(odd_labels[x][-1]))

"""
                directory_names = []
                for root, directories, files in os.walk(data_dir):
                    for directory in directories:
                        directory_names.append(directory)

                # Calculate metrics for k-NN
                unique_labels = np.unique(odd_labels)

                # Initialize variables to store per-class TP, FP, FN
                true_positives = np.zeros(len(unique_labels)+1)
                false_positives = np.zeros(len(unique_labels)+1)
                false_negatives = np.zeros(len(unique_labels)+1)
                i = 0
                k=0
                # Iterate over each unique label
                for label in unique_labels:
                    true_positive = 0
                    false_positive = 0
                    false_negative = 0
                    i+=1
                    j=0
                    # Compare each example's true label and k-NN prediction
                    for true_label:
                        j=j+1
                        if true_label == label:
                            if predicted_label == label:
                                true_positive += 1
                            else:
                                false_positive += 1
                            k = k+1
                        else:
                            k = k+1
                            break

                    false_negative = j-true_positive

                    # Update the respective lists for each class
                    true_positives[i] = true_positive
                    false_positives[i] = false_positive
                    false_negatives[i] = false_negative

                # Calculate precision and recall for each class
                precision_per_class = []
                recall_per_class = []
                for i, label in enumerate(unique_labels):
                    if true_positives[i] + false_positives[i] == 0:
                        precision = 0  # Handle the case of zero in the denominator
                    else:
                        precision = true_positives[i] / (true_positives[i] + false_positives[i])

                    if true_positives[i] + false_negatives[i] == 0:
                        recall = 0  # Handle the case of zero in the denominator
                    else:
                        recall = true_positives[i] / (true_positives[i] + false_negatives[i])

                    precision_per_class.append(precision)
                    recall_per_class.append(recall)

                # Print the precision and recall per class
                for i, label in enumerate(unique_labels):
                    print(f'Label {label}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}')

                # Calculate overall accuracy
                accuracy_knn = sum(1 for a, p in zip(odd_labels, odd_labels_knn) if a == p) / len(odd_labels)
                print(f"Accuracy: {accuracy_knn}")
                for i in range(0, len(odd_labels_knn)):
                    print(f"Image {i + 1}: Predicted Label = {odd_labels_knn[i]} , Original Label = {odd_labels[i]}")
"""



if __name__ == "__main__":
    main()