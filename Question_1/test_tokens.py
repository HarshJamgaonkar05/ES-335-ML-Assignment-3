import torch

checkpoint = torch.load('Models/linux_kernel_best_model.pth', map_location='cpu')
word_to_idx = checkpoint['word_to_idx']
idx_to_word = checkpoint['idx_to_word']
vocab = checkpoint['vocab']

test_words = ['static', 'int', 'struct', 'return']

print("Testing tokenization:")
for word in test_words:
    if word in word_to_idx:
        idx = word_to_idx[word]
        print(f"'{word}' -> index {idx} -> '{idx_to_word[idx]}'")
    else:
        print(f"'{word}' -> NOT IN VOCAB")

print("\n\nChecking special tokens:")
special = ['<START>', '<END>', '<UNK>', '<NEWLINE>']
for token in special:
    if token in word_to_idx:
        print(f"'{token}' -> index {word_to_idx[token]}")
    else:
        print(f"'{token}' -> NOT IN VOCAB")

print(f"\n\nContext length: {checkpoint['context_length']}")
print(f"Vocab size: {len(vocab)}")
