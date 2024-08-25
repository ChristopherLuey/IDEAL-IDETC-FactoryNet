import os
import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
)

MODEL = "gpt-4o-mini"  # Model name

d = {}
# Open and read classes_name.txt
with open("filtered_classes4.txt", 'r') as file:
    for line in file:
        label, qid = line.strip().split('\t')
        d[label] = qid


def create_gpt_prompt_for_labels(labels):
    prompt = ""
    prompt += "\n".join([f"Data {i + 1}: {label}" for i, label in enumerate(labels)])
    return prompt

def get_gpt_responses_for_batches(batches):
    combined_hierarchy = {}
    for batch in batches:
        prompt = create_gpt_prompt_for_labels(batch)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in data organization. You are given a list of items in batches. "
                        "Your task is to identify the hierarchy among these items and organize them into a structured tree, "
                        "where broader categories are at the top and more specific items are nested under them. "
                        "If an item does not belong under any other item, it should be at the top level. "
                        "You are strictly limited to using only the items provided in the list. "
                        "Do not make parents or any other node something that isn't provided in the list. "
                        "Additionally, ensure that the hierarchy connects with items from previous batches to form a complete and unified structure. "
                        "Please return the hierarchy in a structured format like JSON. \n"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL,
            response_format={"type": "json_object"},

        )
        batch_response = chat_completion.choices[0].message.content.strip()
        batch_hierarchy = json.loads(batch_response)
        
        # Merge the batch hierarchy into the combined hierarchy
        combined_hierarchy = merge_hierarchies(combined_hierarchy, batch_hierarchy)
    
    return combined_hierarchy

def merge_hierarchies(base_hierarchy, new_hierarchy):
    """Merge new_hierarchy into base_hierarchy, combining any overlapping keys."""
    for key, value in new_hierarchy.items():
        if key in base_hierarchy:
            if isinstance(base_hierarchy[key], dict) and isinstance(value, dict):
                base_hierarchy[key] = merge_hierarchies(base_hierarchy[key], value)
        else:
            base_hierarchy[key] = value
    return base_hierarchy

def parse_hierarchy(hierarchy, hierarchy_list, parent=None):
    """Recursively parse the hierarchy and generate the 'Term	subclassOf	Class' structure."""
    for term, children in hierarchy.items():
        if parent:
            try:
                hierarchy_list.append(f"{d[term]}\tsubclassOf\t{d[parent]}")
            except:
                print(f"Unknown term {term} or {parent}")
        #else:
            #hierarchy_list.append(f"{d[term]}\tsubclassOf\tNone")  # Top-level terms
        if isinstance(children, dict):
            parse_hierarchy(children, hierarchy_list, term)

def split_into_batches(data, batch_size):
    """Splits the data into smaller batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    file_path = 'filtered_classes4.txt'
    json_output_file = 'hierarchy_output1.json'
    txt_output_file = 'hierarchy_output1.txt'
    batch_size = 50  # Adjust batch size as needed

    labels = list(d.keys())
    label_batches = list(split_into_batches(labels, batch_size))
    combined_hierarchy = get_gpt_responses_for_batches(label_batches)
    
    # Save the combined hierarchy as JSON
    with open(json_output_file, 'w') as json_file:
        json.dump(combined_hierarchy, json_file, indent=4)
    
    hierarchy_list = []
    parse_hierarchy(combined_hierarchy, hierarchy_list)
    
    # Save the hierarchy to a text file in the desired format
    with open(txt_output_file, 'w') as txt_file:
        txt_file.write("\n".join(hierarchy_list))
    
    print(f"Hierarchy has been successfully saved to {json_output_file} and {txt_output_file}.")

if __name__ == "__main__":
    main()