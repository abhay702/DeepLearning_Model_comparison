import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and parse the Intent.json file
def load_intents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['intents']

# Prepare data for training
def prepare_data(intents):
    texts = []
    labels = []
    for intent in intents:
        for pattern in intent['text']:
            texts.append(pattern)
            labels.append(intent['intent'])
    return texts, labels

# Preprocess text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load and preprocess data
intents = load_intents('Intent.json')
texts, labels = prepare_data(intents)
texts = [preprocess_text(text) for text in texts]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Convert labels to categorical
num_classes = len(label_encoder.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

# Build the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get unique classes in the test set
unique_classes = np.unique(y_test)
class_names = label_encoder.classes_[unique_classes]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted', labels=unique_classes)
recall = recall_score(y_test, y_pred_classes, average='weighted', labels=unique_classes)
f1 = f1_score(y_test, y_pred_classes, average='weighted', labels=unique_classes)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, labels=unique_classes, target_names=class_names))

# Print class distribution in test set
print("\nClass Distribution in Test Set:")
for class_id, class_name in zip(unique_classes, class_names):
    count = np.sum(y_test == class_id)
    print(f"{class_name}: {count}")

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

# Function to predict intent
def predict_intent(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Test the model with some example queries
example_queries = [
    "Hello, how are you?",
    "What time is it?",
    "Tell me a joke",
    "Thank you for your help",
    "Can you prove you're self-aware?"
]

print("\nTesting the model with example queries:")
for query in example_queries:
    predicted_intent = predict_intent(query)
    print(f"Query: '{query}'\nPredicted Intent: {predicted_intent}\n")

# Save the model and necessary components
model.save('intent_classification_model.h5')
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)
with open('label_encoder.json', 'w') as f:
    json.dump(list(label_encoder.classes_), f)
np.save('max_sequence_length.npy', max_sequence_length)

print("Model and components saved. You can load them later for predictions.")