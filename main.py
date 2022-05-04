# importing modules
from tkinter import Tk, FALSE, Menu, Text
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model
import json
import random

# importing training data
df = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/training.csv")
# loading model
chatbot= load_model("chatbot")
# loading responses
responses = json.load(open("C:/Users/ASUS/OneDrive/Desktop/responses.json", "r"))

# fitting TfIdfVectorizer with training data to preprocess inputs
df["patterns"] = df["patterns"].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
vectorizer.fit(df["patterns"])

# fitting LabelEncoder with target variable(tags) for inverse transformation of predictions
le = LabelEncoder()
le.fit(df["tags"])

# transforming input and predicting intent


def predict_tag(inp_str):
    inp_data_tfidf = vectorizer.transform([inp_str.lower()]).toarray()
    predicted_proba = chatbot.predict(inp_data_tfidf)
    encoded_label = [np.argmax(predicted_proba)]
    predicted_tag = le.inverse_transform(encoded_label)[0]
    return predicted_tag


# defining chat function


def start_chat():
    print("---------------  AI Chat bot  ---------------")
    print("Ask any queries...")
    print("I will try to understand you and reply...")
    print("Type EXIT to quit...")
    while True:
        inp = input("Ask anything... : ")
        if inp == "EXIT":
            break
        else:
            if inp:
                tag = predict_tag(inp)
                response = random.choice(responses[tag])
                print("Response... : ", response)
            else:
                pass

# Import the library
from tkinter import *

root = Tk()

root.title("Chat Bot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

main_menu = Menu(root)

# Create the submenu
file_menu = Menu(root)

# Add commands to submenu
file_menu.add_command(label="New..")
file_menu.add_command(label="Save As..")
file_menu.add_command(label="Exit")
main_menu.add_cascade(label="File", menu=file_menu)
#Add the rest of the menu options to the main menu
main_menu.add_command(label="Edit")
main_menu.add_command(label="Quit")
root.config(menu=main_menu)

chatWindow = Text(root, bd=1, bg="black",  width="50", height="8", font=("Arial", 23), foreground="#00ffff")
chatWindow.place(x=6,y=6, height=385, width=370)

messageWindow = Text(root, bd=0, bg="black",width="30", height="4", font=("Arial", 23), foreground="#00ffff")
messageWindow.place(x=128, y=400, height=88, width=260)

scrollbar = Scrollbar(root, command=chatWindow.yview, cursor="star")
scrollbar.place(x=375,y=5, height=385)

Button = Button(root, text="Send", command=, width="12", height=5,
                    bd=0, bg="#0080ff", activebackground="#00bfff",foreground='#ffffff',font=("Arial", 12))
Button.place(x=6, y=400, height=88)


root.mainloop()



# calling chat function to start chatting


start_chat()




