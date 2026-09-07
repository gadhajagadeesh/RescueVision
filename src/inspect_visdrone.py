import os

dataset_path = "datasets/VisDrone2019-DET-train"

if os.path.exists(dataset_path):
    print("VisDrone training dataset found!")
else:
    print("VisDrone training dataset is not available yet.")