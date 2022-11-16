# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:30:25 2022

@author: Jesús Alexandro Hernández Rivera
"""

from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import io
import requests

def open_file(path: str) -> str:
    content = ""
    with open(path, "r") as f:
        content = f.readlines()
    return " ".join(content)

def get_csv_from_url(url:str) -> pd.DataFrame:
    try:
        s = requests.get(url).content
        print("Extracción exitosa")
        return pd.read_csv(io.StringIO(s.decode('utf-8')))
    except:
        print("Sin conexión a internet\nSe leyó de un archivo local")
        return pd.read_csv("melb_data.csv")

url = "https://raw.githubusercontent.com/Chutzi/Datasets/main/melb_data.csv"
df = get_csv_from_url(url)
df = df["Suburb"]

import csv
df.to_csv('text.txt', sep=" ", 
          quoting=csv.QUOTE_NONE, escapechar=" ")

all_words = ""
texto = open_file("text.txt") 
words = texto.rstrip().split(" ")

Counter(" ".join(words).split()).most_common(10)
# looping through all incidents and joining them to one text, to extract most common words
for arg in words:
    tokens = arg.split()
    all_words += " ".join(tokens) + " "

print(all_words)
wordcloud = WordCloud(
    background_color="white", min_font_size=5
).generate(all_words)

# print(all_words)
# plot the WordCloud image
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# plt.show()
plt.savefig("img/word_cloud.png")
plt.close()