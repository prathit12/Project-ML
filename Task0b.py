import numpy as np
import pymongo


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
    even_images = list(collection.find({"_id": {"$mod": [2, 0]}}))
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
        images = list(collection.find({"_id": {"$mod": [2, 0]}, "label": label}))
        image_fd_list = [np.array(doc["fd"]) for doc in images]

        # We are performing the calculations to find the inherent dimensionality of the even-numbered images of the specific label.
        num_components = find_pca_comps(image_fd_list, threshold / 100)

        print(f"({i}) Inherent Dimensionality of {label} is: \033[1m{num_components}\033[0m")
        i = i + 1


if __name__ == "__main__":
    import pymongo
    import numpy as np

    task0()
