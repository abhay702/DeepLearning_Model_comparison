# Import necessary libraries
import os
import json
import numpy as np
import tensorflow as tf
import re

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle
from collections import Counter

# Download NLTK data files (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize variables
lemmatizer = WordNetLemmatizer()
ignore_letters = ["!", "?", ",", ".", "'"]

# Function to normalize text
def normalize_text(text):
    # Convert to lowercase, remove punctuation, strip whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# Function to parse the .txt file
def parse_txt_data(file_path):
    intents = []
    with open(file_path, "r", encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Q: "):
                # Extract the question
                question = line[3:].strip()
                i += 1
                # Expect an answer next
                if i < len(lines) and lines[i].strip().startswith("A: "):
                    answer = lines[i].strip()[3:].strip()
                    i += 1
                else:
                    answer = "I'm sorry, I don't have this information."
                # Normalize the answer for comparison
                normalized_answer = normalize_text(answer)
                # Check if the intent already exists
                existing_intent = next(
                    (
                        intent for intent in intents 
                        if normalize_text(intent["responses"][0]) == normalized_answer
                    ), 
                    None
                )
                if existing_intent:
                    existing_intent["patterns"].append(question)
                else:
                    intents.append(
                        {
                            "tag": f"tag_{len(intents)}",
                            "patterns": [question],
                            "responses": [answer],
                        }
                    )
            else:
                i += 1
    return {"intents": intents}

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence = sentence.lower()
    sentence = ''.join([char for char in sentence if char not in ignore_letters])
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to predict the class
def predict_class(sentence, model):
    # Clean up the sentence
    sentence_words = clean_up_sentence(sentence)
    # Convert to a sentence string
    sentence_str = ' '.join(sentence_words)
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([sentence_str])
    # Pad the sequence
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)
    # Predict
    res = model.predict(padded_seq)[0]
    # Get top results
    ERROR_THRESHOLD = 0.5  # Adjusted confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # If no results above threshold, return empty list
    if not results:
        return []
    return_list = []
    for r in results:
        return_list.append(
            {"intent": label_encoder.inverse_transform([r[0]])[0], "probability": str(r[1])}
        )
    return return_list

# Function to get the response
def get_response(intents_list, intents_json, user_input):
    if intents_list:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = i["responses"][0]
                break
        else:
            result = "I'm sorry, I don't have this information. You can ask for question assistance."
    else:
        # Use retrieval-based approach
        result = retrieve_response(user_input, intents_json)
    return result

# Function to retrieve response using TF-IDF and cosine similarity
def retrieve_response(user_input, intents_json):
    all_questions = []
    all_responses = []
    for intent in intents_json["intents"]:
        for pattern in intent["patterns"]:
            all_questions.append(pattern)
            all_responses.append(intent["responses"][0])

    # Vectorize the questions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_questions + [user_input])

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get the index of the most similar question
    most_similar_idx = cosine_similarities.argmax()
    confidence = cosine_similarities[most_similar_idx]

    # Set a threshold for similarity
    SIMILARITY_THRESHOLD = 0.2  # Adjust as needed
    if confidence > SIMILARITY_THRESHOLD:
        return all_responses[most_similar_idx]
    else:
        return "I'm sorry, I don't have this information. You can ask for question assistance."

# Load and parse your dataset
data = parse_txt_data("dialogs.txt")

if "intents" in data and data["intents"]:
    # Initialize lists
    sentences = []
    labels = []
    classes = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Clean and tokenize each pattern
            sentence_words = clean_up_sentence(pattern)
            sentence = ' '.join(sentence_words)
            sentences.append(sentence)
            labels.append(intent["tag"])
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    classes = sorted(list(set(classes)))

    # Count samples per class
    label_counts = Counter(labels)
    print("Number of samples per class:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Create the tokenizer with oov_token
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences
    max_sequence_length = max([len(seq) for seq in sequences])
    max_sequence_length = 20  # Fixed max length for consistency

    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    y = to_categorical(encoded_labels)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check if the model already exists
    if os.path.exists("chatbot_model.h5"):
        # Load the existing model
        model = load_model("chatbot_model.h5")
        # Load tokenizer, label_encoder, max_sequence_length
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        with open("label_encoder.pickle", "rb") as handle:
            label_encoder = pickle.load(handle)
        with open("max_sequence_length.json", "r") as f:
            max_sequence_length = json.load(f)["max_sequence_length"]
    else:
        # Define embedding dimension
        embedding_dim = 128  # Adjusted dimension

        # Create the model
        model = Sequential()
        model.add(
            Embedding(
                input_dim=len(word_index) + 1,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
            )
        )
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dropout(0.5))
        model.add(Dense(len(classes), activation="softmax"))

        # Compile the model
        adam = Adam(learning_rate=0.0005)  # Reduced learning rate
        model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

        # Define EarlyStopping callback
        callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        )

        # Train the model
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=16,
            verbose=1,
            callbacks=[callback],
        )

        # Save tokenizer and label_encoder
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("label_encoder.pickle", "wb") as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("max_sequence_length.json", "w") as f:
            json.dump({"max_sequence_length": max_sequence_length}, f)

        # Save the model
        model.save("chatbot_model.h5", overwrite=True)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# User interaction
print(
    "\nWelcome to the INFOASSIST Bot! Start typing to ask me a question. Type 'quit' to exit.\n"
)

while True:
    print("\nYou: ", end="")
    user_input = input()
    if user_input.lower() == "quit":
        break
    ints = predict_class(user_input, model)
    res = get_response(ints, data, user_input)
    print("ChatBot: ", res)

    # Ask the user if the answer was correct
    print("\nWas the answer correct? (yes/no): ", end="")
    feedback = input()

    # If the answer was not correct, list out all questions that are most similar to the user's question
    if feedback.lower() == "no":
        print("\nHere are some other questions that might help:\n")

        # Get all questions from your dataset
        all_questions = [
            pattern for intent in data["intents"] for pattern in intent["patterns"]
        ]

        # Calculate tf-idf vectors for the user's question and all other questions
        tfidf_matrix = vectorizer.fit_transform([user_input] + all_questions)

        # Calculate cosine similarity between the user's question and all other questions
        cosine_similarities = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()

        # Get the top 5 most similar questions
        top_question_indices = cosine_similarities.argsort()[:-6:-1]
        for i, index in enumerate(top_question_indices):
            print(f"{i + 1}. {all_questions[index]}")

        print(
            "\nEnter the question number from above for which you want the answer to."
        )

        user_choice = input().strip()
        if user_choice.isdigit():
            selected_index = int(user_choice) - 1
            if 0 <= selected_index < len(top_question_indices):
                selected_question = all_questions[top_question_indices[selected_index]]
                selected_ints = predict_class(selected_question, model)
                selected_res = get_response(selected_ints, data, selected_question)
                print(f"ChatBot: {selected_res}")
            else:
                print("Invalid number. Please try again.")
        elif user_choice.lower() == "keywords":
            print(
                "\nType the main keywords of your question if your required question is not listed above:"
            )
        else:
            print("Invalid input. Please try again.")
