# WildVision
The project implements an animal detection and classification system using Histogram of Oriented Gradients (HoG) for feature extraction and a Support Vector Machine (SVM) for classification. The system can detect and classify multiple animal species based on trained models.

# Features

Detects and classifies animals from images with high accuracy.

Uses HoG for feature extraction, ensuring robust feature representation.

Trains an SVM model for efficient classification.

Shows test images with predicted labels and accuracy in the title.

Provides visualization for better interpretability of predictions.

# Installation

Ensure you have Python installed. Then, install the required dependencies using:

pip install -r requirements.txt

# Dataset Structure

Organize the dataset in the following format:

selected_dataset/
│── animal_1/
│   ├── image1.jpg
│   ├── image2.jpg
│── animal_2/
│   ├── image1.jpg
│   ├── image2.jpg
│...

Each subfolder should represent a different animal category, containing images of that species.

# Usage

Training and Evaluation

Run the following command to train the model and evaluate its performance:

python animal_detection.py

The script will output the model's accuracy and generate a confusion matrix for further analysis.

# Testing with New Images

To classify a new image, update the image path in animal_detection.py and execute:

python animal_detection.py

The system will display the image along with the predicted label and accuracy overlayed.

# Output

![image](https://github.com/user-attachments/assets/2c80f67d-c557-4452-8b4b-d326cd87dc9e)
![image](https://github.com/user-attachments/assets/4d9fe7c6-dc10-4d6e-b352-902d77a69f0a)

# Dependencies

OpenCV

NumPy

Scikit-learn

Matplotlib

Seaborn

Scikit-image
