import os
import csv
import json
import requests
import pickle
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
)

MODEL = "gpt-4o"

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as file:
            return pickle.load(file)
    return {}

def save_cache(cache, cache_file):
    with open(cache_file, 'wb') as file:
        pickle.dump(cache, file)

def load_existing_classes(classes_name_file):
    existing_classes = {}
    if os.path.exists(classes_name_file):
        with open(classes_name_file, 'r') as file:
            for line in file:
                label, qid, name = line.strip().split('\t')
                existing_classes[label] = (qid, name)
    return existing_classes

def query_wikidata(label, cache):
    if label in cache:
        return cache[label]
    
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={label}&language=en&format=json"
    response = requests.get(url)
    data = response.json()

    if data['search']:
        qid = data['search'][0]['id']  # Q-identifier
        name = data['search'][0]['label']  # WikiData label name
        cache[label] = (qid, name)
        return qid, name

    cache[label] = (None, None)  # Cache negative results to avoid repeated queries
    return None, None

def clean_labels_batch(batched_labels):
    prompt = ""
    for i, labels in enumerate(batched_labels):
        prompt += f"Set {i + 1}: {labels}\n"
    print(prompt)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You will be provided with multiple sets of human-labeled data related to manufacturing images. "
                    "For each set, your task is to clean and combine the labels into the smallest possible number of labels to represent the image concepts. "
                    "When labels are even remotely similar or related, consolidate them into a single label that best represents the content. "
                    "Prioritize the use of labels that are well-known, widely recognized and used. However, keep it somewhat specific. "
                    "Ensure that each concept is represented by only one label, but retain multiple labels if they represent different important concepts in the image set. "
                    "Do not have repeated labels. And reduce the number of labels used."
                    "Accuracy and specificity are crucial, so choose or generate only one label that best and most precisely describe each concept, while ensuring all major concepts are included. "
                    "If a label is too vague or general, generate a more specific term. "
                    "Make sure all the labels are relevant to manufacturing so and choose labels so they can be searched on Wikidata."
                    "Follow these formatting rules closely: return the combined labels for each set as a Python list, formatted exactly like this example: "
                    "['milling machine'] "
                    "['bolt'] "
                    "['wire cutter'] "
                    "Each set should be on a new line, with each list in the same format. "
                    "The output should be plain text (not Python formatted) so that it can be directly parsed by Python's eval() function. "
                    "Ensure that each individual list appears on a separate line, corresponding to each original set, and do not make it a list of lists."
                    "Be sure to escape any single quotes in the labels using a backslash so that Python's eval() function can parse. "
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=MODEL,
    )

    cleaned_labels_batch = chat_completion.choices[0].message.content
    print(cleaned_labels_batch)

    # Ensure the output is a list of Python lists, remove any code formatting
    cleaned_labels_batch = cleaned_labels_batch.replace("```python", "").replace("```", "").strip()

    parsed_labels = []
    for line in cleaned_labels_batch.splitlines():
        line = line.strip()
        if line:
            try:
                parsed_labels.append(eval(line))
            except (SyntaxError, ValueError):
                print(f"Skipping problematic line: {line}")

    # Ensure parsed_labels is never None
    cleaned_labels_batch = parsed_labels if parsed_labels else []
    
    return cleaned_labels_batch

def process_csv_files(csv_paths, existing_classes, cache):
    batched_labels = []
    unknown_labels = []
    cleaned_labels_for_json = {}

    for csv_path in csv_paths:
        labels = []
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                label = row[0]
                labels.append(label)
        batched_labels.append(labels)

    cleaned_labels_batch = clean_labels_batch(batched_labels)
    wikidata_classes = []
    wikidata_names = []
    label_to_wikidata = {}

    for csv_path, labels in zip(csv_paths, cleaned_labels_batch):
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        cleaned_labels_for_json[base_name] = labels

        for label in labels:
            # Check if the label already exists in classes_name.txt or in the processed batch
            if label in existing_classes:
                qid, name = existing_classes[label]
            elif label in label_to_wikidata:  # Check in the current batch to avoid duplication
                qid = label_to_wikidata[label]
                name = existing_classes.get(label, (qid, label))[1]  # Fetch name from existing classes if available
            else:
                qid, name = query_wikidata(label, cache)
                if qid:
                    existing_classes[label] = (qid, name)
                else:
                    unknown_labels.append(label)

            if qid:
                # Only add to wikidata_classes if it's not already added
                if (name, qid) not in wikidata_classes:
                    wikidata_classes.append((name, qid))
                # Map label to QID
                label_to_wikidata[label] = qid
                wikidata_names.append((label, qid, name))

    return wikidata_classes, wikidata_names, unknown_labels, label_to_wikidata, cleaned_labels_for_json

def main():
    data_dir = 'hackathon/data'
    json_path = 'cleaned_labels.json'
    classes_file = 'classes.txt'
    classes_name_file = 'classes_name.txt'
    unknown_wikidata_file = 'unknown_wikidata.txt'
    cache_file = 'wikidata_cache.pkl'  # Cache file to store the WikiData queries

    image_labels = {}

    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            try:
                if os.stat(json_path).st_size > 0:
                    image_labels = json.load(json_file)
            except json.JSONDecodeError:
                print(f"{json_path} is empty or contains invalid JSON. Starting fresh.")
                image_labels = {}

    # Load existing classes from classes_name.txt
    existing_classes = load_existing_classes(classes_name_file)

    # Load the WikiData cache from the file
    cache = load_cache(cache_file)

    csv_paths = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            base_name = os.path.splitext(file_name)[0]
            if base_name in image_labels:
                continue
            
            csv_path = os.path.join(data_dir, file_name)
            csv_paths.append(csv_path)
            
            if len(csv_paths) >= 1000:  # Adjust this batch size as needed
                wikidata_classes, wikidata_names, unknown_labels, label_to_wikidata, cleaned_labels_for_json = process_csv_files(csv_paths, existing_classes, cache)
                
                # Store the cleaned labels in the JSON
                image_labels.update(cleaned_labels_for_json)

                with open(json_path, 'w') as json_file:
                    json.dump(image_labels, json_file, indent=4)
                
                with open(classes_file, 'a') as file:
                    for label, qid in wikidata_classes:
                        file.write(f"{label}\t{qid}\n")
                
                with open(classes_name_file, 'a') as file:
                    for label, qid, name in wikidata_names:
                        file.write(f"{label}\t{qid}\t{name}\n")

                with open(unknown_wikidata_file, 'a') as file:
                    for label in unknown_labels:
                        file.write(f"{label}\n")
                
                # Store the dictionary linking cleaned labels to WikiData classes
                with open('label_to_wikidata.json', 'w') as json_file:
                    json.dump(label_to_wikidata, json_file, indent=4)
                
                # Save the cache to a file
                save_cache(cache, cache_file)

                print(f"Processed and updated JSON, classes, and unknown labels for batch: {csv_paths}")
                
                csv_paths = []

    if csv_paths:
        wikidata_classes, wikidata_names, unknown_labels, label_to_wikidata, cleaned_labels_for_json = process_csv_files(csv_paths, existing_classes, cache)
        
        # Store the cleaned labels in the JSON
        image_labels.update(cleaned_labels_for_json)

        with open(json_path, 'w') as json_file:
            json.dump(image_labels, json_file, indent=4)
        
        with open(classes_file, 'a') as file:
            for label, qid in wikidata_classes:
                file.write(f"{label}\t{qid}\n")
        
        with open(classes_name_file, 'a') as file:
            for label, qid, name in wikidata_names:
                file.write(f"{label}\t{qid}\t{name}\n")

        with open(unknown_wikidata_file, 'a') as file:
            for label in unknown_labels:
                file.write(f"{label}\n")
        
        # Store the dictionary linking cleaned labels to WikiData classes
        with open('label_to_wikidata.json', 'w') as json_file:
            json.dump(label_to_wikidata, json_file, indent=4)
        
        # Save the cache to a file
        save_cache(cache, cache_file)

        print(f"Processed and updated JSON, classes, and unknown labels for final batch: {csv_paths}")

    print("Data sanitization complete. Cleaned labels, classes, names, and unknown labels saved.")

if __name__ == "__main__":
    main()
