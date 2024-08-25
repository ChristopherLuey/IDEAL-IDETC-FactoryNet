import os
import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
)

MODEL = "gpt-4-turbo"  # Model name

def load_classes(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def create_gpt_prompt_for_labels(labels):
    prompt = ""
    prompt += "\n".join([f"Label {i + 1}: {label}" for i, label in enumerate(labels)])
    return prompt

def split_into_batches(data, batch_size):
    """Splits the data into smaller batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def get_gpt_responses_for_batches(batches):
    all_responses = []
    for batch in batches:
        prompt = create_gpt_prompt_for_labels(batch)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You will be provided with a list of WikiData labels that are being considered for classifying manufacturing images. "
                        "Your task is to evaluate each label and determine whether it is broad, relevant, and useful for categorizing tools, equipment, processes, or materials specifically related to manufacturing. "
                        "If a label is appropriate for classifying common manufacturing tools, equipment, or processes—such as 'nut,' 'bolt,' or 'saw'—respond with 'Keep.' "
                        "If a label is irrelevant, overly specific, historical, academic, related to a location, or associated with a non-manufacturing context, respond with 'Remove.' "
                        "For example, remove labels like 'Taking Measurements - The Artist Copying a Cast in the Hall of the National Gallery of Ireland' because it is related to art and not manufacturing. "
                        "Remove labels like 'Stone chisels from Wood Walton, Huntingdonshire' because it is a specific historical artifact, not a general manufacturing tool. "
                        "Remove labels like 'Golden Spike National Historical Park' because it is a location, not a manufacturing-related label. "
                        "Remove labels like 'Meat grinders and molecular epidemiology: two supermarket outbreaks of Escherichia coli O157:H7 infection' because it is an academic study not related to manufacturing. "
                        "Remove labels like 'Jigsaws-a preserved ability in semantic dementia' because it is related to a medical condition, not manufacturing. "
                        "Ensure that your 'Keep' or 'Remove' response is on a new line and corresponds directly to the order of the labels provided. "
                        "Only respond with 'Keep' or 'Remove.' Focus on whether the label would be effective, relevant, and accurate for classifying general manufacturing-related images. "
                        "Retain labels that are broad and applicable across various aspects of manufacturing, such as common tools, machinery, and processes, while removing those that are out of scope."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL,
        )
        batch_responses = chat_completion.choices[0].message.content.strip().split('\n')
        all_responses.extend(batch_responses)
    return all_responses

def filter_labels_based_on_gpt_response(labels, responses):
    filtered_labels = []
    for label, response in zip(labels, responses):
        if response.lower() == 'keep':
            filtered_labels.append(label)
    return filtered_labels

def save_filtered_labels(filtered_labels, output_file):
    with open(output_file, 'w') as file:
        for label in filtered_labels:
            file.write(label + '\n')

def main():
    file_path = 'duplicate_remove2.txt'
    output_file = 'filtered_classes4.txt'
    batch_size = 20  # Adjust batch size as needed

    labels = load_classes(file_path)
    label_batches = list(split_into_batches(labels, batch_size))
    gpt_responses = get_gpt_responses_for_batches(label_batches)
    filtered_labels = filter_labels_based_on_gpt_response(labels, gpt_responses)
    save_filtered_labels(filtered_labels, output_file)

    print(f"Filtered labels saved to {output_file}")

if __name__ == "__main__":
    main()
