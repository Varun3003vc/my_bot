import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.models import save_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/training.csv")

# preprocessing training data
df["patterns"] = df["patterns"].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
training_data_tfidf = vectorizer.fit_transform(df["patterns"]).toarray()

# preprocessing target variable(tags)
le = LabelEncoder()
training_data_tags_le = pd.DataFrame({"tags": le.fit_transform(df["tags"])})
training_data_tags_dummy_encoded = pd.get_dummies(training_data_tags_le["tags"]).to_numpy()

# creating DNN-
chatbot = Sequential()
chatbot.add(Dense(10, input_shape=(len(training_data_tfidf[0]),)))
chatbot.add(Dense(8))
chatbot.add(Dense(8))
chatbot.add(Dense(6))
chatbot.add(Dense(len(training_data_tags_dummy_encoded[0]), activation="softmax"))
chatbot.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# fitting DNN
chatbot.fit(training_data_tfidf, training_data_tags_dummy_encoded, epochs=100, batch_size=32)

# saving model file
save_model(chatbot, "chatbot")


