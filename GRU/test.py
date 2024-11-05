import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def load_intents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['intents']

def prepare_data(intents):
    texts = []
    labels = []
    for intent in intents:
        for pattern in intent['text']:
            texts.append(pattern)
            labels.append(intent['intent'])
    return texts, labels

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def build_gru_model(vocab_size, embedding_dim, max_sequence_length, num_classes):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
        GRU(128, return_sequences=True),
        Dropout(0.5),
        GRU(64),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def predict_intent(text, model, tokenizer, label_encoder, max_sequence_length):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def evaluate_model(model, X_test, y_test, tokenizer, label_encoder, max_sequence_length, intents):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
    label_names = label_encoder.inverse_transform(unique_labels)

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='macro', labels=unique_labels, zero_division=1)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted', labels=unique_labels, zero_division=1)

    cm = confusion_matrix(y_true, y_pred_classes, labels=unique_labels)
    cr = classification_report(y_true, y_pred_classes, labels=unique_labels, target_names=label_names, output_dict=True, zero_division=1)

    response_times = []
    bleu_scores = []
    for intent in intents:
        for pattern in intent['text']:
            start_time = time.time()
            predicted_intent = predict_intent(pattern, model, tokenizer, label_encoder, max_sequence_length)
            end_time = time.time()
            response_times.append(end_time - start_time)
            bleu_scores.append(sentence_bleu([pattern.split()], predicted_intent.split()))

    avg_response_time = np.mean(response_times)
    avg_bleu_score = np.mean(bleu_scores)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": cr,
        "avg_response_time": avg_response_time,
        "avg_bleu_score": avg_bleu_score,
        "unique_labels": unique_labels,
        "label_names": label_names
    }

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_history.png')
    plt.close()

def plot_confusion_matrix(cm, label_names, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, 
                yticklabels=label_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    intents = load_intents('Intent.json')
    texts, labels = prepare_data(intents)
    texts = [preprocess_text(text) for text in texts]

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Tokenize texts
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

    # Convert labels to categorical
    num_classes = len(label_encoder.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build the GRU model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128
    gru_model = build_gru_model(vocab_size, embedding_dim, max_sequence_length, num_classes)

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = gru_model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    print("\nEvaluating the GRU model...")
    evaluation_results = evaluate_model(gru_model, X_test, y_test_cat, tokenizer, label_encoder, max_sequence_length, intents)

    # Print evaluation results
    print("\nGRU Model Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Macro Precision: {evaluation_results['precision_macro']:.4f}")
    print(f"Macro Recall: {evaluation_results['recall_macro']:.4f}")
    print(f"Macro F1-score: {evaluation_results['f1_macro']:.4f}")
    print(f"Weighted Precision: {evaluation_results['precision_weighted']:.4f}")
    print(f"Weighted Recall: {evaluation_results['recall_weighted']:.4f}")
    print(f"Weighted F1-score: {evaluation_results['f1_weighted']:.4f}")
    print(f"Average Response Time: {evaluation_results['avg_response_time']:.4f} seconds")
    print(f"Average BLEU Score: {evaluation_results['avg_bleu_score']:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(np.argmax(y_test_cat, axis=1), np.argmax(gru_model.predict(X_test), axis=1), 
                                labels=evaluation_results['unique_labels'], 
                                target_names=evaluation_results['label_names'], 
                                zero_division=1))

    # Plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix'], evaluation_results['label_names'], "GRU")

    # Plot training history
    plot_training_history(history, "GRU")

    print("\nTraining history and confusion matrix plots have been saved.")

    # Test the model with example queries
    example_queries = [
        "Hello, how are you?",
        "What time is it?",
        "Tell me a joke",
        "Thank you for your help",
        "Can you prove you're self-aware?"
    ]

    print("\nTesting the GRU model with example queries:")
    for query in example_queries:
        predicted_intent = predict_intent(query, gru_model, tokenizer, label_encoder, max_sequence_length)
        print(f"Query: '{query}'\nPredicted Intent: {predicted_intent}\n")

    # Save the model and necessary components
    gru_model.save('gru_intent_classification_model.h5')
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer.to_json(), f)
    with open('label_encoder.json', 'w') as f:
        json.dump(list(label_encoder.classes_), f)
    np.save('max_sequence_length.npy', max_sequence_length)

    print("GRU model and components saved. You can load them later for predictions.")

    # Interactive testing
    print("\nEnter a message to test the GRU chatbot (type 'quit' to exit):")
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            break
        predicted_intent = predict_intent(message, gru_model, tokenizer, label_encoder, max_sequence_length)
        print(f"ChatBot: Predicted Intent: {predicted_intent}")