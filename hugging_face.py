from datasets import DatasetDict,load_from_disk
from huggingface_hub import login
login()  
dataset = load_from_disk("translated_dataset") 
new_dataset = DatasetDict({
    "train": dataset  
})

new_dataset.push_to_hub("Kiran189/CognitiveLabs")
# This code is used to push the translated dataset to Hugging Face Hub
# The dataset is saved in the "translated_dataset" folder
# Link to colab notebook: https://colab.research.google.com/drive/1aNu1a6KP6JZmW3C1wclz9tjN6skn_32Y?usp=sharing