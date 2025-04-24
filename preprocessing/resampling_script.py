import os
import pandas as pd

from preprocessing.smote import Smote
from preprocessing.tomek import TomekLinks


"""
A script to apply SMOTE and Tomek Links resampling techniques to the training data.
It would save the resampled data in separate folders for each technique.
"""


base_dir = "dataset"
subfolders = {
    "smote": os.path.join(base_dir, "smote_resample"),
    "tomek": os.path.join(base_dir, "tomek_resample")
}

for folder in subfolders.values():
    os.makedirs(folder, exist_ok=True)

X_train = pd.read_csv(os.path.join(base_dir, "X_train.csv")).values
y_train = pd.read_csv(os.path.join(base_dir, "y_train.csv")).values.ravel()


print("Applying SMOTE")
smote = Smote(k_neighbours=5, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

pd.DataFrame(X_smote).to_csv(os.path.join(subfolders["smote"], "X_train.csv"), index=False)
pd.DataFrame(y_smote).to_csv(os.path.join(subfolders["smote"], "y_train.csv"), index=False)
print("SMOTE resampling complete")



print("Applying Tomek Links")
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)

pd.DataFrame(X_tomek).to_csv(os.path.join(subfolders["tomek"], "X_train.csv"), index=False)
pd.DataFrame(y_tomek).to_csv(os.path.join(subfolders["tomek"], "y_train.csv"), index=False)
print("Tomek Links resampling complete.")


