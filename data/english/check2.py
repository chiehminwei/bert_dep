with open('predictions.txt') as f, open('oracle.txt') as ff:
	for i, (l, ll) in enumerate(zip(f, ff)):
		if l.split('\t')[0] != ll.split('\t')[0] and not l.split('\t')[0].startswith('[UNK]') and l.split('\t')[0] != '"': 
			print(i)
			break
			