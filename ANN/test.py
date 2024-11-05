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

# Function to parse the Intent.json file
def parse_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    intents = data.get('intents', [])
    return intents

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
        for intent in intents_json:
            if intent["intent"] == tag:
                result = np.random.choice(intent["responses"])
                break
    else:
        result = "I'm sorry, I don't have this information. You can take the question assistance."
    return result

# Load and parse your dataset
intents = parse_json_data("Intent.json")

# Prepare data for training
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

# Create training data
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

# Ensure training is a two-dimensional array
training = np.array(training, dtype=object)

# Now you can safely access train_x and train_y
train_x = list(training[:, 0])  # feature set
train_y = list(training[:, 1])  # target outcomes

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
    res = get_response(ints, intents)
    print("ChatBot: ", res)

