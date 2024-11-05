# Import necessary libraries
import os
import json
import keras
import numpy as np
import tensorflow

# from tensorflow.keras.models import Sequential
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout

# from keras.optimizers import SGD
import nltk

# nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Function to convert a sentence into a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


# Function to predict the class
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Function to get the response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                # Sort the responses by probability
                sorted_responses = sorted(
                    i["responses"], key=lambda x: x[1], reverse=True
                )
                # Choose the response with the highest probability
                result = sorted_responses[0]
                break
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

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Create training data
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
    #Here we have cereated one hot encoding of the vector 
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    # Ensure training is a two-dimensional array
    training = np.array(training, dtype=object)

    # Check if training is not a 2D array
    if len(training.shape) == 1:
        # Reshape training to be a 2D array if it's currently 1D
        training = np.array([training.tolist()])

    # Now you can safely access train_x and train_y
    train_x = list(training[:, 0]) #feature set
    train_y = list(training[:, 1]) #Target outcomes

    # Check if the model already exists
    if os.path.exists("chatbot_model.h5"):
        # Load the existing model
        model = load_model("chatbot_model.h5")
    else:
        # Create the model
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        # model.add(Dense(16, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation="softmax"))

        # Compile the model
        adam = keras.optimizers.Adam(learning_rate=0.001)
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
            np.array(train_x),
            np.array(train_y),
            epochs=1000,
            batch_size=10,
            verbose=1,
            callbacks=callback,
        )

        # Save words and classes
        with open("words.json", "w") as f:
            json.dump(words, f)
        with open("classes.json", "w") as f:
            json.dump(classes, f)
        with open("intents.json", "w") as f:
            json.dump(data, f)

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
            print("Invalid input. Please try again.")