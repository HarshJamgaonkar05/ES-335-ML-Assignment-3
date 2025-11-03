"""
Next-Word Prediction Streamlit App
ES-335 ML Assignment 3 - Task 1.4

Features:
- Multiple trained models (Shakespeare, Linux Kernel)
- Temperature control for sampling randomness
- Configurable generation parameters
- OOV (Out-of-Vocabulary) handling
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path


class NextWordMLP(nn.Module):
    """MLP-based next-word prediction model"""
    def __init__(self, vocab_size, embedding_dim, context_length, hidden_dim, activation='relu', dropout=0.5):
        super(NextWordMLP, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = context_length * embedding_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1)
        
        h1 = self.activation(self.fc1(embedded))
        h1 = self.dropout(h1)
        
        h2 = self.activation(self.fc2(h1))
        h2 = self.dropout(h2)
        
        output = self.fc3(h2)
        return output


@st.cache_resource
def load_model(model_path):
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:

        checkpoint = torch.load(model_path, map_location='cpu')
        

        vocab = checkpoint['vocab']
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        context_length = checkpoint['context_length']
        embedding_dim = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_dim']
        activation = checkpoint.get('activation', 'relu')
        

        model = NextWordMLP(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            context_length=context_length,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=0.0  
        )
        

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        

        return {
            'model': model,
            'vocab': vocab,
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'context_length': context_length,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'activation': activation,
            'vocab_size': len(vocab)
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



def generate_text(model_info, seed_text, num_words, temperature=1.0, is_code=False):
    model = model_info['model']
    word_to_idx = model_info['word_to_idx']
    idx_to_word = model_info['idx_to_word']
    context_length = model_info['context_length']
    
    model.eval()
    

    if is_code:
        words = seed_text if isinstance(seed_text, list) else seed_text.split()
    else:
        words = seed_text.lower().split()
    
    generated = words.copy()
    

    for _ in range(num_words):
        # Get context window
        context = generated[-context_length:] if len(generated) >= context_length else generated
        

        context_indices = [word_to_idx.get(w, word_to_idx.get('<UNK>', 0)) for w in context]
        

        if len(context_indices) < context_length:
            start_idx = word_to_idx.get('<START>', 0)
            context_indices = [start_idx] * (context_length - len(context_indices)) + context_indices
        

        context_tensor = torch.LongTensor([context_indices])
        

        with torch.no_grad():
            output = model(context_tensor)
            logits = output[0] / temperature
            probs = F.softmax(logits, dim=0)          
            predicted_idx = torch.multinomial(probs, 1).item()
        next_word = idx_to_word[predicted_idx]
        if next_word == '<END>':
            break     
        generated.append(next_word) 
    return ' '.join(generated)



def main():
    st.set_page_config(
        page_title="Next-Word Prediction",
        page_icon="",
        layout="wide"
    )
    
    st.title(" Next-Word Prediction with MLP")
    st.markdown("### ES-335 ML Assignment 3 - Task 1.4")
    
    st.markdown("""
    Next 'k' word prediction using trained MLP models.
    """)
    
    st.sidebar.header(" Configuration")
    
    available_models = {
        "Shakespeare (Natural Language)": "Models/shakespeare_best_model.pth",
        "Linux Kernel (Code)": "Models/linux_kernel_best_model.pth"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
    )
    
    model_path = available_models[selected_model_name]
    is_code = "Linux" in selected_model_name
    
    with st.spinner(f"Loading {selected_model_name}..."):
        model_info = load_model(model_path)
    
    if model_info is None:
        st.error("Failed to load model. Please check that model files exist in the Models/ directory.")
        return
    
    st.sidebar.success(" Model loaded!")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander(" Model Details"):
        st.write(f"**Vocabulary Size:** {model_info['vocab_size']:,}")
        st.write(f"**Embedding Dim:** {model_info['embedding_dim']}")
        st.write(f"**Hidden Dim:** {model_info['hidden_dim']}")
        st.write(f"**Context Length:** {model_info['context_length']}")
        st.write(f"**Activation:** {model_info['activation'].upper()}")
    
    st.sidebar.markdown("---")
    
    st.sidebar.header(" Generation Controls")
    
    num_words = st.sidebar.slider(
        "Number of words to generate",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
    )
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    st.sidebar.markdown("---")
    
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Input")
        
        if is_code:
            default_examples = [
                "struct task_struct *",
                "if (",
                "int ret =",
                "static void"
            ]
            placeholder = "struct task_struct *"
        else:
            default_examples = [
                "to be or not to",
                "what is the",
                "i am",
                "the king of"
            ]
            placeholder = "to be or not to"
        
        st.markdown("**Quick Examples:**")
        example_cols = st.columns(len(default_examples))
        for i, example in enumerate(default_examples):
            if example_cols[i].button(f"`{example}`", key=f"ex_{i}"):
                st.session_state.seed_text = example
        
        st.markdown("---")
        
        seed_text = st.text_area(
            "Enter seed text:",
            value=st.session_state.get('seed_text', placeholder),
            height=100,
            help="Enter a few words to start. The model will continue from here."
        )
        
        generate_button = st.button(" Generate Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader(" Output")
        
        if generate_button:
            if not seed_text.strip():
                st.warning("Please enter some seed text.")
            else:
                with st.spinner("Generating..."):
                    # Generate text
                    generated = generate_text(
                        model_info,
                        seed_text,
                        num_words,
                        temperature,
                        is_code
                    )
                    
                    # Display result
                    st.markdown("**Generated Text:**")
                    st.text_area(
                        "Result",
                        value=generated,
                        height=200,
                        label_visibility="collapsed"
                    )
                    
                    word_count = len(generated.split())
                    st.caption(f" Generated {word_count} words total")
        else:
            st.info(" Configure settings and click **Generate Text** to start")
    

    
    st.markdown("---")
    


if __name__ == "__main__":
    main()
