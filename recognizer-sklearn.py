import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Read training images
data = pd.read_csv("trainingsample.csv").values
training_pixels = data[0:, 1:]
training_labels = data[0:, 0]

# Train
clf = DecisionTreeClassifier()
clf.fit(training_pixels, training_labels)

# Read validation images
validation_data = pd.read_csv("validationsample-bigger.csv").values
validation_pixels = validation_data[0:, 1:]
validation_labels = validation_data[0:, 0]

# Predict the labels (digits)
predicted_labels = clf.predict(validation_pixels)

# Measure the accuracy of the model
labels_count = len(validation_labels)
matches_count = 0

for i in range (0, labels_count):
    matches_count += 1 if predicted_labels[i] == validation_labels[i] else 0

print ("Accuracy: {0:.02%}".format(matches_count / labels_count)) 

# Plot an image, see if it matches the predicted digit
image = validation_pixels[8]
image.shape = (28, 28)
pt.imshow(255 - image, cmap = 'gray')
print("Digit: ", clf.predict([validation_pixels[8]])[0])
pt.show()

