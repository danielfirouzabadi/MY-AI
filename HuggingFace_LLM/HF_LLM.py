from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

emotions = load_dataset("emotion")
emotions
print(emotions)

train_ds = emotions["train"] #look into train from the emotion dataset 
print(train_ds)

print(len(train_ds)) #length of train dataset
print(train_ds[0]) #accesss a single exmaple by its index
print(train_ds.column_names) #access column names
print(train_ds.features) #access features
print(train_ds[:5]) #access 5 rows
print(train_ds["text"][:5]) #get the full column by name

emotions.set_format(type="pandas") #convert to pandas dataframe for visualization 
df=emotions["train"][:]
print(df.head())

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row) #use int2str to change label integers to add coloumn w/ corresponding labelnames

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

#using Matplotlib for visualizing class distribution
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()