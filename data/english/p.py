tags = set()
with open('all') as f:
	for line in f:
		tokens=line.split()
		for token in tokens:
			#word = token.split('|')[0]
			tag = token.split('|')[-1]
			tags.add(tag)

print(len(tags))



unknown map to -1