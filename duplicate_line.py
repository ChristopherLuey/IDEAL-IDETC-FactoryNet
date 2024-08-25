file_path = "classes.txt"

# Read the lines from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Track duplicates
seen = set()
duplicates = []

# Identify duplicates while preserving order
unique_lines = []
for line in lines:
    if line in seen:
        duplicates.append(line)
    else:
        seen.add(line)
        unique_lines.append(line)

# Write the unique lines back to the file
with open("duplicate_remove2.txt", 'w') as file:
    file.writelines(unique_lines)

# Print out the duplicate lines
print("Duplicate lines:")
for duplicate in duplicates:
    print(duplicate.strip())