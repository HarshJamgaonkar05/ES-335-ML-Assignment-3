import re
import pickle
import urllib.request
from collections import Counter
import numpy as np

def download_dataset(url, filename):
    print(f"Downloading {filename}")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def preprocess_shakespeare(input_file, output_prefix):    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
        
    text = re.sub(r"\s+", " ", text)
    text = re.sub('[^a-zA-Z0-9 \\.]', '', text)
    text = text.lower()
    
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    all_words = []
    for sentence in sentences:
        words = sentence.split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    
    vocab_size = len(word_counts)
    print(f"\nVocabulary size: {vocab_size}")
    
    most_common = word_counts.most_common(10)
    least_common = word_counts.most_common()[-10:]
    
    print("\n10 Most frequent words:")
    for word, count in most_common:
        print(f"  {word}: {count}")
    
    print("\n10 Least frequent words:")
    for word, count in least_common:
        print(f"  {word}: {count}")
    
    min_freq = 2
    
    vocab = ['<UNK>', '<START>', '<END>'] 
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab.append(word)
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"\nFinal vocabulary size (with min_freq={min_freq}): {len(vocab)}")
    

    X_data = []
    y_data = []
    
    context_length = 5  
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 2:  
            continue
        
        
        words = ['<START>'] + words + ['<END>']
        
        
        word_indices = []
        for word in words:
            if word in word_to_idx:
                word_indices.append(word_to_idx[word])
            else:
                word_indices.append(word_to_idx['<UNK>'])
        
        
        for i in range(1, len(word_indices)):
            context_start = max(0, i - context_length)
            context = word_indices[context_start:i]
            
            if len(context) < context_length:
                context = [word_to_idx['<START>']] * (context_length - len(context)) + context
            
            target = word_indices[i]
            
            X_data.append(context)
            y_data.append(target)
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\nTraining samples created: {len(X_data)}")
    print(f"Context shape: {X_data.shape}")
    print(f"Target shape: {y_data.shape}")


    data = {
        'X': X_data,
        'y': y_data,
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'context_length': context_length,
        'vocab_stats': {
            'vocab_size': len(vocab),
            'total_words': len(all_words),
            'most_common': most_common,
            'least_common': least_common,
            'min_freq': min_freq
        }
    }
    
    
    with open(f'{output_prefix}_processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    with open(f'{output_prefix}_vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f"{word}\n")
    
    print(f"\nSaved processed data to {output_prefix}_processed.pkl")
    print(f"Saved vocabulary to {output_prefix}_vocab.txt")
    
    return data

def preprocess_linux_kernel(input_file, output_prefix):

    print("\n=== Processing Linux Kernel Code Dataset ===")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    all_tokens = []
    for line in lines:
        tokens = line.split()
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    
    vocab_size = len(token_counts)
    print(f"\nVocabulary size: {vocab_size}")
    
    most_common = token_counts.most_common(10)
    least_common = token_counts.most_common()[-10:]
    
    print("\n10 Most frequent tokens:")
    for token, count in most_common:
        print(f"  {repr(token)}: {count}")
    
    print("\n10 Least frequent tokens:")
    for token, count in least_common:
        print(f"  {repr(token)}: {count}")
    
    
    min_freq = 3  
    
    vocab = ['<UNK>', '<START>', '<END>', '<NEWLINE>'] 
    for token, count in token_counts.most_common():
        if count >= min_freq:
            vocab.append(token)
    
    word_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_word = {idx: token for token, idx in word_to_idx.items()}
    
    print(f"\nFinal vocabulary size (with min_freq={min_freq}): {len(vocab)}")
    
    X_data = []
    y_data = []
    
    context_length = 10  
    
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:  
            continue
             
        tokens = ['<START>'] + tokens + ['<END>']
        
        token_indices = []
        for token in tokens:
            if token in word_to_idx:
                token_indices.append(word_to_idx[token])
            else:
                token_indices.append(word_to_idx['<UNK>'])
        
        for i in range(1, len(token_indices)):
            # Get context
            context_start = max(0, i - context_length)
            context = token_indices[context_start:i]
        
            if len(context) < context_length:
                context = [word_to_idx['<START>']] * (context_length - len(context)) + context
            
            target = token_indices[i]
            
            X_data.append(context)
            y_data.append(target)
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\nTraining samples created: {len(X_data)}")
    print(f"Context shape: {X_data.shape}")
    print(f"Target shape: {y_data.shape}")
    
    
    data = {
        'X': X_data,
        'y': y_data,
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'context_length': context_length,
        'vocab_stats': {
            'vocab_size': len(vocab),
            'total_tokens': len(all_tokens),
            'most_common': most_common,
            'least_common': least_common,
            'min_freq': min_freq
        }
    }
    
    
    with open(f'{output_prefix}_processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    
    with open(f'{output_prefix}_vocab.txt', 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")
    
    print(f"\nSaved processed data to {output_prefix}_processed.pkl")
    print(f"Saved vocabulary to {output_prefix}_vocab.txt")
    
    return data

if __name__ == "__main__":
    SHAKESPEARE_URL = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
    LINUX_URL = "https://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt"
    
    download_dataset(SHAKESPEARE_URL, "shakespeare_input.txt")
    download_dataset(LINUX_URL, "linux_input.txt")

    shakespeare_data = preprocess_shakespeare("shakespeare_input.txt", "shakespeare")
    linux_data = preprocess_linux_kernel("linux_input.txt", "linux_kernel")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
