import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and parse the Intent.json file
def parse_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    intents = data.get('intents', [])
    return intents

# Prepare data for training
def prepare_data(intents):
    words = []
    classes = []
    documents = []
    ignore_letters = ["!", "?", ",", "."]

    for intent in intents:
        for pattern in intent["text"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["intent"]))
            if intent["intent"] not in classes:
                classes.append(intent["intent"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return words, classes, documents

# Create training data
def create_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    np.random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return train_x, train_y

# Build and train the model
def build_and_train_model(train_x, train_y, test_x, test_y):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(train_y[0]), activation="softmax")
    ])

    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.00001, patience=20, verbose=1, mode="auto",
        baseline=None, restore_best_weights=True
    )

    history = model.fit(
        np.array(train_x), np.array(train_y),
        epochs=1000, batch_size=10, verbose=1,
        validation_split=0.2, callbacks=[early_stopping]
    )

    return model, history

# Simplified evaluate_model function
def evaluate_model(model, test_x, test_y, history):
    # Make predictions
    y_pred = model.predict(np.array(test_x))
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_y, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # Print class distribution
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_true, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Class {i}: {count}")

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    intents = parse_json_data("Intent.json")
    words, classes, documents = prepare_data(intents)
    train_x, train_y = create_training_data(words, classes, documents)

    # Split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    # Build and train the model
    model, history = build_and_train_model(train_x, train_y, test_x, test_y)

    # Evaluate the model
    evaluate_model(model, test_x, test_y, history)

    print("\nEvaluation completed. Check the generated plots for visual results.")

    # Optional: Save the model
    model.save("intent_classification_model.h5")
    print("Model saved as 'intent_classification_model.h5'")