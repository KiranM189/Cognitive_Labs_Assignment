# Name : Kiran M - Applying for the Big Data Intern Role
# Here I am using Modal to parallelly distribute the function of translating the dataset from English to Kannada
# Modal is a serverless framework that allows you to run Python code in parallel across multiple containers
import modal
from deep_translator import GoogleTranslator
from datasets import load_dataset

app = modal.App("dataset-translator")

# Using list to define required columns for the dataset
required_columns = ["question", "choices", "hint", "task", "grade", "subject", "topic", "category", "skill", "solution"]

# Creating an image for the translation function with necessary libraries installed
# which inturn is used for creating instances of containers for parallel execution
translator_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "deep-translator", "datasets"
)

# This contains the Modal function wrapped with the decorator to translate the dataset batchwise
# which runs parallelly across multiple containers
# Here the maximum number of containers is set to 100 and each container recieves a batch of 100 rows (refer below calculation)
# This means that 100 containers are concurrently exucted to carry out the translation with specific timeout
@app.function(image=translator_image, max_containers=100, timeout=600)
def translate_batch(batch):
    translator = GoogleTranslator(source="en", target="kn")

    def translate(text):
        try:
            return translator.translate(text)
        except Exception as e:
            return f"[ERROR: {str(e)}]"

    #initializing a dictionary to hold the translated batch    
    translated_batch = { col: [] for col in required_columns }
    for col in required_columns:
        if col == "choices": # handling choices separately as they are lists of lists
            translated_batch[col] = [
                [translate(choice) for choice in choice_list] for choice_list in batch[col]
            ]
        else:
            translated_batch[col] = [translate(item) for item in batch[col]]
    return translated_batch


# Local entrypoint to the function (similar to if __name__ == "__main__")
@app.local_entrypoint()
def main():
    print("Loading dataset...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="train[:10000]")

    # Here each batch contains 100 rows
    # So the size of batches is also 100 (i.e. 10000 rows / 100 rows per batch = 100 batches)
    # Each batch is assigned to one container for parallel execution 
    batch_size = 100
    batches = []

    # Dividing the dataset into multiple batches of specific batch size
    print(f"Dividing dataset into batches of size {batch_size}...")
    for i in range(0, len(dataset), batch_size):
        batch = {}
        for col in required_columns:
            batch[col] = dataset[col][i:i + batch_size] 
        batches.append(batch)

    # Applying the translation function to each batch in parallel
    # The map fuction will take iterate over the batch inside batches and 
    # will apply the translate_batch function to each batch
    print("Calling translation function for each batch in parallel...")
    translated_batches = list(translate_batch.map(batches))



    # Aggregating the translated results form batches into a single dictionary
    # Initializing the dictionary to hold the translated columns
    print("Translation done!!!")
    translated_columns = {col: [] for col in required_columns}
    for result in translated_batches:
        for col in required_columns:
            translated_columns[col].extend(result[col])

    # Adding the translated columns to the dataset with "_kannada" suffix to the column names
    for col in required_columns:
        dataset = dataset.add_column(col + "_kannada", translated_columns[col])


    # Removing the old columns 
    required_columns.append("lecture") # Here lecture column is also removed as its not translated due to api call limits
    dataset = dataset.remove_columns(required_columns)

    # Saving the translated dataset locally which later is used to push to Hugging Face
    dataset.save_to_disk("translated_dataset")
    print("Saved translated dataset to translated_dataset folder")
