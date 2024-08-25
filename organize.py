qids = []
with open("results/classes.txt", 'r') as file:
    for line in file:
        label, qid = line.strip().split('\t')
        qids.append(qid)
        

with open("results/classes2.txt", 'w') as file:
    for qid in qids:
        file.write(f"{qid}\n")
