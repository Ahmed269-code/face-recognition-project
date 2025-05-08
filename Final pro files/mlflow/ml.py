
import os
import glob
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
from sklearn.decomposition import PCA

from deepface import DeepFace

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import os
import pandas as pd

# Define the base directory where images are stored
base_dir = "D:\\downloads\\lfw_faces data+me"

# List all image files recursively
image_paths = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(root, file))

# Extract person names from filenames
persons = []
for path in image_paths:
    filename = os.path.basename(path)  # e.g., "Aaron_Eckhart_0001.jpg"
    person_name = "_".join(filename.split("_")[:-1])  # Removes "_0001.jpg"
    persons.append(person_name)
    

# Create a DataFrame
df = pd.DataFrame({
    "person": persons,
    "path": image_paths
})


# remove people with fewer than 5 images
filtered_dataset = df.groupby("person").filter(lambda x: len(x) >= 10)

# limit each person to max 30 images
def limit_images(group):
    return group.sample(n=min(len(group), 30), random_state=42)

df = filtered_dataset.groupby("person").apply(limit_images).reset_index(drop=True)


import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define augmentation pipeline
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Resize(100, 100)  # optional: to ensure uniform size
])

# Create lists for new (augmented) data
augmented_data = {
    "person": [],
    "path": [],
    "image": []
}

# Loop through each image and apply augmentation
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img = cv2.imread(row["path"])
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply augmentation
    augmented = augmenter(image=img)
    aug_img = augmented['image']

    # Save the augmented image and person label
    augmented_data["person"].append(row["person"])
    augmented_data["path"].append("augmented")  # or save later if needed
    augmented_data["image"].append(aug_img)

# Create DataFrame for augmented data
aug_df = pd.DataFrame(augmented_data)

# Add original image array column for training
df["image"] = df["path"].apply(lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))

# Merge original and augmented data
final_df = pd.concat([df, aug_df], ignore_index=True)


def get_embedding_from_array(img_array):
    try:
        embedding = DeepFace.represent(
            img_path=img_array,        # Pass image array instead of path
            model_name="VGG-Face",
            enforce_detection=False
        )[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None
    
final_df["embedding"] = final_df["image"].apply(get_embedding_from_array)

final_df = final_df.dropna(subset=["embedding"]).reset_index(drop=True)


label_mapping = {name: idx for idx, name in enumerate(final_df["person"].unique())}
final_df["label"] = final_df["person"].map(label_mapping)

embeddings = final_df["embedding"].tolist()

labels = final_df["label"].tolist()

assert len(embeddings) == len(labels), "Mismatch between embeddings and labels!"

X = np.array(embeddings)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# بدء تجربة MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SVM Face Recognition")

# التجارب اللي هنختبر بيها
param_grid = [
    {"C": 1, "gamma": "scale"},
    {"C": 10, "gamma": "scale"},
    {"C": 100, "gamma": "auto"}
]

for params in param_grid:
    with mlflow.start_run():
        svm = SVC(C=params["C"], gamma=params["gamma"], kernel='rbf', probability=True)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # تسجيل القيم
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # حفظ النموذج
        mlflow.sklearn.log_model(svm, "svm_model")

        print(f"Parameters: {params} => Accuracy: {acc:.4f}")

