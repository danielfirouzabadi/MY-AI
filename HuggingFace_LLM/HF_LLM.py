

from datasets import load_dataset
emotions = load_dataset("emotion")
emotions
print(emotions)

train_ds = emotions["train"] #look into train from the emotion dataset 
print(train_ds)

print(len(train_ds)) #length of train dataset
print(train_ds[0]) #accesss a single exmaple by its index
print(train_ds.column_names) #access column names
print(train_ds.features) #access features