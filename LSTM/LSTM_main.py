# Import necessary libraries
import os
import json
import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle


# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Initialize variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ["!", "?", ",", "."]

# Function to parse the .txt file
def parse_txt_data(file_path):
    intents = []
    current_tag = ""
    patterns = []
    responses = []
    with open(file_path, "r") as file:
        for line in file:
            # Each line contains a question or an answer, starting with 'Q: ' or 'A: '
            if line.startswith("Q: "):
                # This is a question, so we start a new intent
                current_tag = f"tag_{len(intents)}"
                patterns = [line[3:].strip()]  # Remove 'Q: ' from the start
            elif line.startswith("A: "):
                # This is an answer, so we add it to the current intent
                responses = [line[3:].strip()]  # Remove 'A: ' from the start
                intents.append(
                    {
                        "tag": current_tag,
                        "patterns": patterns,
                        "responses": responses,
                    }
                )
    return {"intents": intents}

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_letters
    ]
    return sentence_words

# Function to predict the class
def predict_class(sentence, model):
    # Clean up the sentence
    sentence_words = clean_up_sentence(sentence)
    # Convert to a sentence string
    sentence_str = " ".join(sentence_words)
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([sentence_str])
    # Pad the sequence
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)
    # Predict
    res = model.predict(padded_seq)[0]
    # Get top results
    ERROR_THRESHOLD = 0.25
    results = [
        [i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {"intent": label_encoder.inverse_transform([r[0]])[0], "probability": str(r[1])}
        )
    return return_list

# Function to get the response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = i["responses"][0]
                break
        else:
            result = "I'm sorry, I don't have this information. You can take the question assistance."
    else:
        result = "I'm sorry, I don't have this information. You can take the question assistance."
    return result

# Load and parse your dataset
data = parse_txt_data("dialogs.txt")

if "intents" in data and data["intents"]:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize each word
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents
            documents.append((word_list, intent["tag"]))
            # Add to classes list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    classes = sorted(list(set(classes)))

    # Collect sentences and labels
    sentences = []
    labels = []

    for doc in documents:
        sentence = " ".join(
            [lemmatizer.lemmatize(w.lower()) for w in doc[0] if w not in ignore_letters]
        )
        sentences.append(sentence)
        labels.append(doc[1])

    # Create the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences
    max_sequence_length = max([len(seq) for seq in sequences])
    max_sequence_length = 20  # You can choose a fixed max length

    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    y = tf.keras.utils.to_categorical(encoded_labels)


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
        # Create the model
        model = Sequential()
        model.add(
            Embedding(
                input_dim=len(word_index) + 1,
                output_dim=128,
                input_length=max_sequence_length,
            )
        )
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(64))
        model.add(Dropout(0.25))
        model.add(Dense(len(classes), activation="softmax"))

        # Compile the model
        adam = Adam(learning_rate=0.001)
        model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

        callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00001,
            patience=20,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )

        # Train the model
        model.fit(
            X,
            y,
            epochs=1000,
            batch_size=10,
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
    res = get_response(ints, data)
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
                selected_res = get_response(selected_ints, data)
                print(f"ChatBot: {selected_res}")
            else:
                print("Invalid number. Please try again.")
        elif user_choice.lower() == "keywords":
            print(
                "\nType the main keywords of your question if your required question is not listed above:"
            )
        else:
            print("Invalid input. Please try again.")
