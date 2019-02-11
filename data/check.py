#import tokenization
#VOCAB_FILE='../data/vocab.txt'
#max_seq_length=128
#tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)
#count=0
with open('test.tsv') as f:
	for line in f:
		words = []
		labels = []
		combs = line.strip().split('\t')
		
		for comb in combs:
			word = comb.split('|')[0]
			words.append(word)
			label = comb.split('|')[-1]
			labels.append(label)
			#print(word + '\t' + label)

		text = ''.join(words)
		if len(text) > 256: 
			print(len(text))
		# tokens = tokenizer.tokenize(text)
		# if len(tokens) > max_seq_length - 2:
  #     		tokens = tokens[0:(max_seq_length - 2)]		

  #     	for word, label in zip(words, labels):
		#     sub_tokens = tokenizer.tokenize(orig_token)
		#     label_ids.extend([label_map[label]] * len(sub_tokens))
		#     # if label_map[label] != 0:
		#     token_start_idxs.append(len(bert_tokens))
		#     bert_tokens.extend(sub_tokens)

#		if len(combs) > 127:
			#count += len(words) - 128
#			combs = combs[:127]
			#print(words)
			#print(len(words))
		
		

		#print('')
#print(count)