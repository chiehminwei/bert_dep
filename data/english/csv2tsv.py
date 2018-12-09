with open('train', 'r') as f:
    lines = f.readlines()
    outs = open('train.tsv', 'w')
    for line in lines:
        line = line.split()
        line = '\t'.join(line) + '\n'
        outs.write(line)
