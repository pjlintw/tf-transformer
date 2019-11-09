
from collections import Counter


input_fn = './src.train.example.txt'

c = Counter()

with open(input_fn, 'r', encoding='utf-8') as f:
    for line_sentence in f:
        for char in line_sentence:
            char = char.strip()
            if char == '' or char == '\t':
                continue
            else:
                c.update([char])

with open('vocab.txt', 'w', encoding='utf-8') as wf:
    for char in c:
        wf.write(char+'\n')


