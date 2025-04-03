import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract HoG features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, 
                        orientations=50, 
                        pixels_per_cell=(32, 32),
                        cells_per_block=(2, 2), 
                        block_norm='L2-Hys',
                        visualize=True)
    return features

# Load dataset
def load_dataset(dataset_path):
    X, y = [], []
    labels = {}  
    label_index = 0  

    for label in sorted(os.listdir(dataset_path)):  
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        if label not in labels:
            labels[label] = label_index
            label_index += 1

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            features = extract_hog_features(img)
            X.append(features)
            y.append(labels[label])

    return np.array(X), np.array(y), labels

# Define dataset path
dataset_path = "selected_dataset"

# Load dataset
X, y, class_labels = load_dataset(dataset_path)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train SVM classifier
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Test model accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Class Labels Mapping:", class_labels)

# Function to test an image
def predict_animal(image_path, model, class_labels):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image!")
        return

    img = cv2.resize(img, (700, 700))  # Increase display size
    features = extract_hog_features(cv2.resize(img, (128, 128)))
    features = np.array(features).reshape(1, -1)  # Reshape for SVM prediction

    prediction = model.predict(features)[0]
    
    # Get the predicted class label
    predicted_label = [key for key, value in class_labels.items() if value == prediction]
    
    if predicted_label:
        label_text = f"Detected Animal: {predicted_label[0]}"
    else:
        label_text = "No matching animal found in the dataset!"
    # Overlay accuracy and label on the image
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    
    # Put the label text and accuracy on the image
    cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, accuracy_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with prediction
    cv2.imshow(f"{label_text} - Accuracy: {accuracy * 100:.2f}%", img)
    cv2.waitKey(0)  # Wait for a key press to close the image
    cv2.destroyAllWindows()

# Example Usage
test_image_path = "tigertest.jpg"  
predict_animal(test_image_path, svm_model, class_labels)

test_image_path = "elephanttest2.jpg"  
predict_animal(test_image_path, svm_model, class_labels)

test_image_path = "liontest.jpg"  
predict_animal(test_image_path, svm_model, class_labels)


test_image_path = "building.jpg"  
predict_animal(test_image_path, svm_model, class_labels)