import numpy as np
import pymongo
import hashlib
import numpy.ma as ma
from IPython.display import display
import torchvision.datasets as datasets
from sklearn.metrics import euclidean_distances, precision_recall_fscore_support, accuracy_score, pairwise_distances
import os
from collections import Counter


def search_similar_images(query_image, t, lsh):
    similar_images = lsh.query(query_image, t)

    top_t_similar = list(similar_images)[:t]  # Convert set to list and retrieve top 't' items

    unique_image_ids = [result[1] for result in top_t_similar]
    similarity_distances = [np.linalg.norm(np.array(result[0]) - np.array(query_image)) for result in top_t_similar]

    overall_count = len(similar_images)

    return unique_image_ids, similarity_distances, overall_count


def find_pca_comps(even_image_fd_list, threshold):
    # Here we are converting our list of FDs into an array
    even_image_fd_array = np.array(even_image_fd_list)

    # Then we want to reshape the data by flattening so that it is easier to perform our PCA operations on the dataset.
    num_rows, num_features = even_image_fd_array.shape[0], even_image_fd_array.shape[1]
    even_image_fd_array = even_image_fd_array.reshape(num_rows, -1)

    # This is the mean calculation of our dataset.
    mean = np.mean(even_image_fd_array, axis=0)

    # Then we proceed to adjust our data to the center to prep the dataset for our PCA operations.
    centered_data = even_image_fd_array - mean

    # Then we calculate the covraiance matrix.
    cov_matrix = np.cov(centered_data, rowvar=False)

    # From our covariance mateix we derive our respect eigen values and eigen vectors.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Then we sort them in descending order.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Then based in these eigen values, we check to see how many eigen values we need to get the cumulative variance of all PCs
    # greater than or equal to the threshold specified.
    cum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = np.argmax(cum_variance >= threshold) + 1

    return num_components


def task0():
    print("------------------------TASK #0A-------------------------")
    print(
        "Here we are going to calculate the Inherent Dimensionality of all the even numbered Image IDs in the Caltech101 dataset.")
    print("We are using Principal Component Analysis (PCA) for this task.\n")

    # This is the threshold provided by the user to compute inherent dimensionality.
    threshold = float(input("Enter threshold in %: "))

    # Establishing connection to our database and it's respective collection.
    database = pymongo.MongoClient("mongodb://localhost:27017/")
    db = database["feature_descriptor_db_2"]
    collection = db["color_moments_2"]

    # Here we are fetching only the even numbered images from the dataset and converting them to be stored in an array.
    even_images = list(collection.find({"image_id": {"$mod": [2, 0]}}))
    even_image_fd_list = [np.array(doc["fd"]) for doc in even_images]

    print(
        f"\nTo check only half of the dataset is being used (because only even Image IDs), shape of dataset: {np.shape(even_image_fd_list)}\n")

    # We are performing the calculations to find the inherent dimensionality of the even-numbered images.
    num_components = find_pca_comps(even_image_fd_list, threshold / 100)

    print(f"\033[1mInherent Dimensionality of even numbered images is: {num_components}\033[0m")

    print("------------------------TASK #0B-------------------------")
    print(
        f"\nWe will now see the inherent dimensionality needed for the images under each label to match the specified threshold of {threshold}%\n")

    # Here we are fetching all the unique labels
    label_list = collection.distinct("label")

    # Initializing for count
    i = 1

    for label in label_list:
        # Fetching the even numbered images from each unique label and storing then in an array.
        images = list(collection.find({"image_id": {"$mod": [2, 0]}, "label": label}))
        image_fd_list = [np.array(doc["fd"]) for doc in images]

        # We are performing the calculations to find the inherent dimensionality of the even-numbered images of the specific label.
        num_components = find_pca_comps(image_fd_list, threshold / 100)

        print(f"({i}) Inherent Dimensionality of {label} is: \033[1m{num_components}\033[0m")
        i = i + 1


def task3():
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

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
            gini = 1.0 - np.sum(probabilities ** 2)
            return gini

        def _best_split(self, X, y):
            m, n = X.shape
            if m <= 1:
                return None, None

            num_parent = list(Counter(y).values())
            best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
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
            return [list(self.label_mapping.keys())[
                        list(self.label_mapping.values()).index(self._predict_tree(sample, self.tree_))] for sample in
                    X]

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

    def personalized_pagerank(graph, even_labels_numeric, target_label, beta, max_iter=50):
        num_nodes = len(graph)
        transition_matrix = calculate_transition_matrix(graph)

        # Initialize PageRank scores
        pagerank_scores = np.ones(num_nodes) / num_nodes

        # Initialize personalization vector
        personalization_vector = np.zeros(num_nodes)
        personalization_vector[even_labels_numeric == target_label] = 1

        for _ in range(max_iter):
            new_pagerank = (1 - beta) * personalization_vector + beta * np.dot(transition_matrix, pagerank_scores)

            # Check for convergence, e.g., using L1 norm
            if np.linalg.norm(new_pagerank - pagerank_scores, 1) < 1e-6:
                break

            pagerank_scores = new_pagerank

        return pagerank_scores

    def calculate_transition_matrix(similarity_matrix):
        # Compute out-degree for each node
        out_degree = np.sum(similarity_matrix, axis=1)

        # Compute transition matrix
        transition_matrix = similarity_matrix / out_degree[:, np.newaxis]

        return transition_matrix



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

        # Calculate precision, recall, and F1 score for each class
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for i in range(len(unique_labels)):
            precision = true_positives[i] / (true_positives[i] + false_positives[i]) if true_positives[i] + \
                                                                                        false_positives[
                                                                                            i] != 0 else 0
            recall = true_positives[i] / (true_positives[i] + false_negatives[i]) if true_positives[i] + \
                                                                                     false_negatives[
                                                                                         i] != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        # Print precision, recall, and F1 score per class
        for i, label in enumerate(unique_labels):
            print(
                f'Label {label}: Precision = {precision_per_class[i]:.4f}, Recall = {recall_per_class[i]:.4f}, F1 Score = {f1_per_class[i]:.4f}')

        # Calculate overall accuracy and print
        accuracy_knn = sum(1 for a, p in zip(odd_labels, predictions) if a == p) / len(odd_labels)
        print(f"Accuracy: {accuracy_knn}")


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
    even_labels = [doc["label"] for doc in even_images]

    data_dir = r'C:\Users\91816\Downloads\CSE-515-Phase1\CSE-515-Phase1\Code\caltech101\101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir)

    while True:
        print(
            "Select a case (1 for M-NN, 2 for Decision Tree Classification, 3 for PPR Classifier) or enter 0 to exit:")
        user_input = input("Enter your choice: ")

        if user_input == "0":
            print("Exiting the program.")
            break

        match user_input:
            case "1":
                m = int(input("Enter the value of 'm' for k-NN: "))
                mnn_classifier = MNNClassifier(m)
                mnn_classifier.fit(even_image_fd_array, even_labels)
                odd_labels_knn = mnn_classifier.predict(odd_image_fd_array)

                for i in range(0, len(odd_labels_knn)):
                    print(f"Image {i + 1}: Predicted Label = {odd_labels_knn[i]} , Original Label = {odd_labels[i]}")

                print_parameters(odd_labels, odd_labels_knn, data_dir)


            case "2":

                even_image_fd_array = perform_pca(even_image_fd_array, 7)
                odd_image_fd_array = perform_pca(odd_image_fd_array, 7)
                print(np.shape(even_image_fd_array))
                tree_classifier = DecisionTreeClassifier(max_depth=54, min_samples_split=20)
                tree_classifier.fit(even_image_fd_array, even_labels)
                predictions = tree_classifier.predict(odd_image_fd_array)

                for i in range(0, len(predictions)):
                    print(f"Image {i + 1}: Predicted Label = {predictions[i]} , Original Label = {odd_labels[i]}")

                print_parameters(odd_labels, predictions, data_dir)

            case "3":

                beta = float(input("Enter random jump probability (Beta value between 0 and 1): "))

                directory_names = []
                for root, directories, files in os.walk(data_dir):
                    for directory in directories:
                        directory_names.append(directory)
                # Load even numbered feature descriptors and their labels from MongoDB
                even_fd = even_image_fd_array
                even_labels = even_labels

                unique_labels = np.unique(even_labels)
                label_mapping = {label: i for i, label in enumerate(unique_labels)}

                # Compute personalized page rank matrix using similarity matrix and labels of even numbered feature descriptors
                even_labels_numeric = [label_mapping[label] for label in even_labels]

                similarity_matrix = np.zeros((len(even_images), len(even_images)))
                for i in range(len(even_labels)):
                    for j in range(len( even_labels)):
                        similarity = cosine_similarity(even_fd[i], even_fd[j])
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity

                pr = personalized_pagerank(similarity_matrix, even_labels_numeric, 'target_label', beta=beta)

                # Load odd numbered feature descriptors and predict their labels using personalized page rank matrix
                odd_fd = odd_image_fd_array
                odd_labels = odd_labels

                predictions = []
                for i, fd in enumerate(odd_fd):
                    similarities = [cosine_similarity(fd, even_fd[j]) for j in range(len(even_fd))]
                    max_sim_index = np.argmax(similarities)
                    predictions.append(even_labels[max_sim_index] if pr[max_sim_index] > 0 else 'unknown')

                for i in range(0, len(predictions)):
                    print(f"Image {i + 1}: Predicted Label = {predictions[i]} , Original Label = {odd_labels[i]}")

                print_parameters(odd_labels, predictions, data_dir)

            case _:
                print("Invalid choice. Please enter 1, 2, 3, or 0 to exit.")


def task4():
    class LSH:
        def __init__(self, L, h, dimensions):
            self.L = L
            self.h = h
            self.dimensions = dimensions
            self.tables = [{} for _ in range(L)]

        def hash_vector(self, v, table_index):
            np.random.seed(table_index)
            random_vectors = np.random.randn(self.h, self.dimensions)
            hashes = np.dot(random_vectors, v)
            return tuple('1' if h > 0 else '0' for h in hashes)

        def add_vector(self, v, image_id):
            for i in range(self.L):
                hash_key = self.hash_vector(v, i)
                if hash_key in self.tables[i]:
                    self.tables[i][hash_key].append((tuple(v), image_id))  # Storing as tuple (vector, image_id)
                else:
                    self.tables[i][hash_key] = [(tuple(v), image_id)]  # Storing as tuple (vector, image_id)

        def create_index(self, vectors, image_ids):
            for vector, image_id in zip(vectors, image_ids):
                self.add_vector(vector, image_id)

        def query(self, v, t):
            results = set()
            for i in range(self.L):
                hash_key = self.hash_vector(v, i)
                if hash_key in self.tables[i]:
                    results.update(self.tables[i][hash_key])

            results = sorted(results, key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(v)))

            return results

    L = int(input("Enter number of layers:"))
    h = int(input("Enter number of hashes per layer:"))
    q_image_id = int(input("Enter Query Image ID:"))
    dimensions = 500

    lsh = LSH(L, h, dimensions)  # Create an instance of the LSH class

    database = pymongo.MongoClient("mongodb://localhost:27017/")
    db = database["feature_descriptor_db_2"]
    collection = db["avgpool_2"]

    even_images = list(collection.find({"image_id": {"$mod": [2, 0]}}))
    even_image_fd_list = [doc["fd"] for doc in even_images]
    even_image_fd_array = np.array(even_image_fd_list)
    even_image_fd_array = even_image_fd_array.reshape(2727, -1)

    even_image_ids = [doc["image_id"] for doc in even_images]

    lsh.create_index(even_image_fd_array.tolist(), even_image_ids)  # Convert to a list of arrays

    qv = collection.find_one({"image_id": q_image_id})
    qv_fd = np.array(qv.get("fd"))
    flattened = qv_fd.reshape(-1)

    t = int(input("Enter the 't' value:\n"))

    similar_ids, distances, overall_count = search_similar_images(flattened, t, lsh)  # Pass lsh as an argument

    data_dir = r'C:\Users\91816\Downloads\CSE-515-Phase1\CSE-515-Phase1\Code\caltech101\101_ObjectCategories'
    dataset = datasets.ImageFolder(root=data_dir)

    for img_id, distance in zip(similar_ids, distances):
        img, label = dataset[img_id]
        display(img)
        print(f"Image ID: {img_id}, Similarity Distance: {distance}")

    print(f"Number of unique images considered: {len(similar_ids)}")
    print(f"Overall number of images considered: {overall_count}")


if __name__ == "__main__":
    task = input("Enter the Task # to be executed:")

    match task:
        case "0":
            task0()
        case "3":
            task3()
        case "4":
            task4()
        case _:
            print("Invalid Task")
