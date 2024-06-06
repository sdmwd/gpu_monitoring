import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate predictions on the test set
predictions = model.predict(test_gen)

# Binarize predictions based on a threshold (0.5)
predicted_labels = (predictions > 0.5).astype(int)

# Actual labels from the test set
true_labels = np.vstack([test_gen[i][1] for i in range(len(test_gen))])

# Function to plot confusion matrix for each class
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Calculate and plot confusion matrix for each class
for i, class_name in enumerate(mlb.classes_):
    cm = confusion_matrix(true_labels[:, i], predicted_labels[:, i])
    plot_confusion_matrix(cm, classes=[f'Not {class_name}', class_name], title=f'Confusion Matrix for {class_name}')
