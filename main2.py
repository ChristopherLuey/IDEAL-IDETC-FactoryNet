from openai import OpenAI

def find_unmatched_classes(classes_name_file, unmatched_file):
    unmatched_entries = []

    # Open and read classes_name.txt
    with open(classes_name_file, 'r') as file:
        for line in file:
            label, qid, name = line.strip().split('\t')
            
            # Compare label and name, case insensitive
            if label.lower() != name.lower():
                unmatched_entries.append(line.strip())

    # Save unmatched entries to unmatched_classes_name.txt
    if unmatched_entries:
        with open(unmatched_file, 'w') as file:
            for entry in unmatched_entries:
                file.write(entry + '\n')
        print(f"Unmatched entries saved to {unmatched_file}.")
    else:
        print("No unmatched entries found.")

find_unmatched_classes('classes_name.txt', 'unmatched_classes_name.txt')


def create_gpt_prompt_for_yes_no(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    prompt = (
        "Please determine if the third column value represents the first column value for each line. "
        "Respond with 'Yes' or 'No' only. Each response should be on a new line, corresponding to the order of the questions below. "
        "Ensure the 'Yes' or 'No' is in the same order as the questions are listed:\n\n"
    )

    for i, line in enumerate(lines, 1):
        columns = line.strip().split('\t')
        if len(columns) >= 3:
            first_value = columns[0]
            third_value = columns[2]
            prompt += f"Set {i}: Does '{third_value}' represent '{first_value}'?\n"

    return prompt

# Initialize the OpenAI client
client = OpenAI(
)

MODEL = "gpt-4o"

# Example usage
file_path = "unmatched_classes_name.txt"  # Update with your file path
gpt_prompt = create_gpt_prompt_for_yes_no(file_path)
print(gpt_prompt)


def remove_no_responses(gpt_output, file_path):
    # Read the lines from the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Split the GPT output into individual lines
    responses = gpt_output.strip().split('\n')

    # Prepare a list to store lines to keep
    lines_to_keep = []

    # Iterate through each line and response
    for i, line in enumerate(lines):
        if i < len(responses):
            response = responses[i].strip()
            if response.lower() == 'no':
                lines_to_keep.append(line)

    # Write the filtered lines back to the file
    with open("unmatched_classes_name_removed.txt", 'w') as file:
        file.writelines(lines_to_keep)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": (
                "You will be provided with pairs of items where the first item is a label derived from WikiData, and the second item is a label intended to represent a specific concept. "
                "These labels will be used to classify manufacturing images, so it's important that the labels accurately represent tools, equipment, and processes related to manufacturing. "
                "Your task is to determine whether the first item accurately represents the second item within the context of manufacturing. "
                "The labels should be broad enough to cover a range of items within a category but still specific enough to be useful for classification. "
                "If the first item is a valid and precise representation of the second item, respond with 'Yes.' If it is not, respond with 'No.' "
                "Each response should be on a new line and correspond directly to the order of the pairs provided. "
                "You may only answer 'Yes' or 'No' for each line. "
                "Ensure that your 'Yes' or 'No' response is in the same order as the pairs are listed. "
                "Consider whether the WikiData label correctly matches the intended concept of the second item, focusing on the accuracy, broadness, and relevance to manufacturing: \n\n"
            )
        },
        {
            "role": "user",
            "content": gpt_prompt
        }
    ],
    model=MODEL,
)
gpt_output = chat_completion.choices[0].message.content

output_file_path = "gpt_output.txt"

# Save the GPT output to the text file
with open(output_file_path, 'w') as file:
    file.write(gpt_output)


remove_no_responses(gpt_output, file_path)


