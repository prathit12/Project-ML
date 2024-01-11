from collections import Counter

import numpy as np
import pymongo
from sklearn.metrics import euclidean_distances, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
import os
import networkx as nx
from sklearn.tree import DecisionTreeClassifier as DT


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    d = dot_product / (magnitude1 * magnitude2)
    return d






class Node:
    def __init__(self, is_leaf=False, class_label=None, threshold=None, index=None):
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.threshold = threshold
        self.index = index
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.label_mapping = None
        self.tree_ = None

    def fit(self, X, y):
        self.label_mapping = {label: i for i, label in enumerate(np.unique(y))}
        y_numeric = np.array([self.label_mapping[label] for label in y])
        self.n_classes_ = len(np.unique(y_numeric))
        self.tree_ = self._grow_tree(X, y_numeric)

    def _gini(self, y):
        counter = Counter(y)
        probabilities = np.array(list(counter.values())) / len(y)
        gini = 1.0 - np.sum(probabilities**2)
        return gini

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = list(Counter(y).values())
        best_gini = 1.0 - sum((num / m)**2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = Counter(y)

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping conditions
        if depth == self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            return Node(is_leaf=True, class_label=np.argmax(np.bincount(y)))

        idx, thr = self._best_split(X, y)

        if idx is not None:
            node = Node(index=idx, threshold=thr)
            indices_left = X[:, idx] <= thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)
            return node
        else:
            return Node(is_leaf=True, class_label=np.argmax(np.bincount(y)))

    def _predict_tree(self, sample, node):
        if node.is_leaf:
            return node.class_label
        if sample[node.index] <= node.threshold:
            return self._predict_tree(sample, node.left)
        else:
            return self._predict_tree(sample, node.right)

    def predict(self, X):
        return [list(self.label_mapping.keys())[list(self.label_mapping.values()).index(self._predict_tree(sample, self.tree_))] for sample in X]


class MNNClassifier:
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            similarities = [cosine_similarity(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(similarities)[::-1][:self.m]  # Sort in reverse to get highest similarities
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            prediction = self.majority_vote(k_nearest_labels)
            predictions.append(prediction)

        return predictions

    def majority_vote(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        return majority_label

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

def print_parameters(odd_labels, predictions, data_dir):
    directory_names = []
    for root, directories, files in os.walk(data_dir):
        for directory in directories:
            directory_names.append(directory)

    unique_labels = np.unique(odd_labels)

    # Initialize variables to store per-class TP, FP, FN
    true_positives = np.zeros(len(unique_labels))
    false_positives = np.zeros(len(unique_labels))
    false_negatives = np.zeros(len(unique_labels))

    # Iterate over each unique label
    for i, label in enumerate(unique_labels):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for true_label, predicted_label in zip(odd_labels, predictions):
            if true_label == label:
                if predicted_label == label:
                    true_positive += 1
                else:
                    false_negative += 1
            elif predicted_label == label:
                false_positive += 1

        # Update the respective lists for each class
        true_positives[i] = true_positive
        false_positives[i] = false_positive
        false_negatives[i] = false_negative

    # Calculate precision and recall for each class
    precision_per_class = []
    recall_per_class = []

    for i in range(len(unique_labels)):
        precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + \
                                                                                    false_positives[
                                                                                        i] != 0 else 0
        recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + \
                                                                                 false_negatives[
                                                                                     i] != 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    # Print the precision and recall per class
    for i, label in enumerate(unique_labels):
        print(
            f'Label {label}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}')

    # Calculate overall accuracy
    accuracy_knn = sum(1 for a, p in zip(odd_labels, predictions) if a == p) / len(odd_labels)
    print(f"Accuracy: {accuracy_knn}")


def main():
    database = pymongo.MongoClient("mongodb://localhost:27017/")
    db = database["feature_descriptor_db_2"]
    collection = db["avgpool_2"]

    # Create index with the vectors and their respective image IDs
    even_images = list(collection.find({"image_id": {"$mod": [2, 0]}}))
    even_image_fd_list = [doc["fd"] for doc in even_images]
    even_image_fd_array = np.array(even_image_fd_list)
    print(np.shape(even_image_fd_array))


    odd_images = list(collection.find({"image_id": {"$mod": [2, 1]}}))
    odd_image_fd_list = np.array([np.array(doc["fd"]) for doc in odd_images])
    odd_image_fd_array = np.array(odd_image_fd_list)
    print(np.shape(odd_image_fd_array))


    odd_labels = [doc["label"] for doc in odd_images]
    even_labels = [doc["label"]for doc in even_images]


    data_dir = r'C:\Users\91816\Downloads\CSE-515-Phase1\CSE-515-Phase1\Code\caltech101\101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir)

    while True:
        print("Select a case (1 for M-NN, 2 for Decision Tree Classification, 3 for PPR Classifier) or enter 0 to exit:")
        user_input = input("Enter your choice: ")

        if user_input == "0":
            print("Exiting the program.")
            break

        match user_input:
            case "1":
                m = int(input("Enter the value of 'm' for k-NN: "))
                mnn_classifier = MNNClassifier(m)
                mnn_classifier.fit(even_image_fd_array, even_labels)
                odd_labels_knn= mnn_classifier.predict(odd_image_fd_array)
                print_parameters(odd_labels, odd_labels_knn, data_dir)

                """
                for i in range(0, len(odd_labels_knn)):
                    print(f"Image {i + 1}: Predicted Label = {odd_labels_knn[i]} , Original Label = {odd_labels[i]}")
                """

            case "2":

                even_image_fd_array = perform_pca(even_image_fd_array, 7)
                odd_image_fd_array = perform_pca(odd_image_fd_array, 7)

                tree_classifier = DecisionTreeClassifier( max_depth=42, min_samples_split=16)
                tree_classifier.fit(even_image_fd_array, even_labels)
                predictions = tree_classifier.predict(odd_image_fd_array)

                print_parameters(odd_labels, predictions, data_dir)

                unique_labels = np.unique(odd_labels)

                # Initialize variables to store per-class TP, FP, FN
                true_positives = np.zeros(len(unique_labels))
                false_positives = np.zeros(len(unique_labels))
                false_negatives = np.zeros(len(unique_labels))

                # Iterate over each unique label
                for i, label in enumerate(unique_labels):
                    true_positive = 0
                    false_positive = 0
                    false_negative = 0

                    for true_label, predicted_label in zip(odd_labels, predictions):
                        if true_label == label:
                            if predicted_label == label:
                                true_positive += 1
                            else:
                                false_negative += 1
                        elif predicted_label == label:
                            false_positive += 1

                    # Update the respective lists for each class
                    true_positives[i] = true_positive
                    false_positives[i] = false_positive
                    false_negatives[i] = false_negative

                # Calculate precision and recall for each class
                precision_per_class = []
                recall_per_class = []

                for i in range(len(unique_labels)):
                    precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + \
                                                                                                false_positives[
                                                                                                    i] != 0 else 0
                    recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + \
                                                                                             false_negatives[
                                                                                                 i] != 0 else 0

                    precision_per_class.append(precision)
                    recall_per_class.append(recall)

                # Print the precision and recall per class
                for i, label in enumerate(unique_labels):
                    print(
                        f'Label {label}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}')

                # Calculate overall accuracy
                accuracy_decision_tree = sum(1 for a, p in zip(odd_labels, predictions) if a == p) / len(
                    odd_labels)
                print(f"Accuracy: {accuracy_decision_tree}")

                # Print named predictions
                for i in range(0, len(predictions)):
                    print(f"Image {i + 1}: Predicted Label = {predictions[i]} , Original Label = {odd_labels[i]}")

            case "3":
                directory_names = []
                for root, directories, files in os.walk(data_dir):
                    for directory in directories:
                        directory_names.append(directory)

                unique_labels = np.unique(even_labels)
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

                unique_labels = np.unique(even_labels)
                label_mapping_1 = {label: idx for idx, label in enumerate(unique_labels)}

                even_labels_numeric = [label_mapping[label] for label in even_labels]
                odd_labels_numeric = [label_mapping_1[label] for label in odd_labels]
                even_labels_numeric = np.array(even_labels_numeric).astype(int)


                even_image_fd_list = perform_pca(even_image_fd_list, 2)
                odd_image_fd_list = perform_pca(odd_image_fd_list, 2)

                unique_labels = np.unique(even_labels_numeric)
                num_images_even, num_label_even = len(even_image_fd_list.tolist()), len(even_labels)

                num_images_odd, num_label_odd = len(even_image_fd_list.tolist()), len(odd_labels)

                distances = euclidean_distances(even_image_fd_list)

                graph = 1 / (1 + distances)
                graph = graph / np.sum(graph, axis=0)

                output = []
                for label in unique_labels:
                    input_images = np.where(even_labels_numeric == label)

                    output.append([label, pageRank(graph, input_images, beta=0.85)])


                predicted_labels = []

                for i in range(len(odd_labels_numeric)):
                    compare = [[output[j][0], output[j][1][i]] for j in range(len(output))]
                    predicted_labels.append(max(compare, key=lambda x: x[1])[0])

                print(predicted_labels)

                # Compute precision, recall, F1-score, and accuracy
                precision, recall, f1, _ = precision_recall_fscore_support(odd_labels_numeric, predicted_labels)
                accuracy = accuracy_score(odd_labels_numeric, predicted_labels)
                for i, label in enumerate(unique_labels):
                    print(f'Label: {label}')
                    print(f'Precision: {precision[i]}')
                    print(f'Recall: {recall[i]}')
                    print(f'F1-score: {f1[i]}\n')
                print(f'Overall accuracy: {accuracy}')


            case _:
                print("Invalid choice. Please enter 1, 2, 3, or 0 to exit.")


if __name__ == "__main__":
    main()