import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.resolve()

class NextWordMLP(nn.Module):
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
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
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
    """Load a trained model with all its metadata"""
    model_path = str(model_path)

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {os.path.abspath(model_path)}")
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
            'vocab_size': len(vocab),
            'training_accuracy': checkpoint.get('best_accuracy', 'N/A'),
            'training_loss': checkpoint.get('best_loss', 'N/A')
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



def handle_oov_words(words, word_to_idx, vocab):
    """Handle Out-of-Vocabulary words"""
    oov_words = []
    oov_positions = []
    
    for i, word in enumerate(words):
        if word not in word_to_idx:
            oov_words.append(word)
            oov_positions.append(i)
    
    return oov_words, oov_positions


def generate_text(model_info, seed_text, num_words, temperature=1.0, is_code=False):
    """Generate text using temperature-based sampling"""
    model = model_info['model']
    word_to_idx = model_info['word_to_idx']
    idx_to_word = model_info['idx_to_word']
    context_length = model_info['context_length']
    vocab = model_info['vocab']
    
    model.eval()
    
    if is_code:
        words = seed_text.split() if isinstance(seed_text, str) else seed_text
    else:
        words = seed_text.lower().split()
    
    oov_words, oov_positions = handle_oov_words(words, word_to_idx, vocab)
    
    generated = words.copy()
    generated_count = 0
    debug_tokens = []
    
    for i in range(num_words):
        context = generated[-context_length:] if len(generated) >= context_length else generated
        
        context_indices = [word_to_idx.get(w, word_to_idx.get('<UNK>', 0)) for w in context]
        
        if len(context_indices) < context_length:
            start_idx = word_to_idx.get('<START>', 0)
            context_indices = [start_idx] * (context_length - len(context_indices)) + context_indices
        
        context_tensor = torch.LongTensor([context_indices])
        
        with torch.no_grad():
            output = model(context_tensor)
            logits = output[0] / temperature
            
            end_token_idx = word_to_idx.get('<END>', -1)
            if end_token_idx >= 0 and i < min(5, num_words // 2):
                logits[end_token_idx] = float('-inf')
            
            probs = F.softmax(logits, dim=0)
            predicted_idx = torch.multinomial(probs, 1).item()
        
        next_word = idx_to_word[predicted_idx]
        debug_tokens.append(f"Step {i+1}: '{next_word}' (idx={predicted_idx})")
        
        if next_word == '<END>':
            debug_tokens.append("Hit <END> token - stopping generation")
            break
        
        if next_word not in ['<START>', '<UNK>', '<NEWLINE>']:
            generated.append(next_word)
            generated_count += 1
        elif next_word == '<NEWLINE>':
            generated.append('\n')
            generated_count += 1
        else:
            debug_tokens.append(f"  Skipped special token: {next_word}")
    
    generated_text = ' '.join(generated)
    oov_info = {
        'oov_words': oov_words,
        'oov_positions': oov_positions,
        'total_oov': len(oov_words),
        'generated_count': generated_count,
        'debug_tokens': debug_tokens
    }
    
    return generated_text, oov_info



def main():
    st.set_page_config(
        page_title="Next-Word Prediction with MLP",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Next-Word Prediction with MLP")
    st.markdown("### ES-335 ML Assignment 3 - Task 1.4")
    
    st.markdown("""
    Generate the next **k words** using trained MLP models with configurable parameters.
    Choose from different model variants with varying architectures and datasets.
    """)
    
    st.sidebar.header("Model Selection")
    
    available_models = {
        "Shakespeare (ReLU)": {
            "path": BASE_DIR / "Models" / "shakespeare_best_model.pth",
            "description": "Trained on Shakespeare's works",
            "type": "text",
            "variant": "base"
        },
        "Shakespeare (Tanh)": {
            "path": BASE_DIR / "Models" / "shakespeare_variant_tanh.pth",
            "description": "Shakespeare model with Tanh activation",
            "type": "text",
            "variant": "tanh"
        },
        "Shakespeare (Large)": {
            "path": BASE_DIR / "Models" / "shakespeare_variant_large.pth",
            "description": "Shakespeare model with larger architecture",
            "type": "text",
            "variant": "large"
        },
        "Linux Kernel (ReLU)": {
            "path": BASE_DIR / "Models" / "linux_kernel_best_model.pth",
            "description": "Trained on Linux kernel source code",
            "type": "code",
            "variant": "base"
        },
        "Linux Kernel (GELU)": {
            "path": BASE_DIR / "Models" / "linux_kernel_variant_gelu.pth",
            "description": "Linux kernel model with GELU activation",
            "type": "code",
            "variant": "gelu"
        },
    }
    
    existing_models = {}
    for name, config in available_models.items():
        if os.path.exists(str(config["path"])):
            existing_models[name] = config
    
    if not existing_models:
        st.error("No trained models found! Please train models first.")
        st.info("Run the training notebook or `train_variants.py` to create models.")
        return
    
    available_models = existing_models
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model Variant",
        list(available_models.keys()),
        help="Choose between different trained model variants"
    )
    
    model_config = available_models[selected_model_name]
    model_path = str(model_config["path"])
    is_code = model_config["type"] == "code"
    
    st.sidebar.caption(f"{model_config['description']}")
    
    with st.sidebar.expander("Compare Model Variants", expanded=False):
        st.markdown("**Available Models:**")
        for model_name, config in available_models.items():
            if model_name == selected_model_name:
                st.markdown(f"**> {model_name}** (selected)")
            else:
                st.markdown(f"- {model_name}")
        
        st.markdown("---")
        st.markdown("""
        **Variant Differences:**
        - **Base Models**: Standard architecture (Context=5, Embed=128)
        - **Tanh Variant**: Uses Tanh activation instead of ReLU
        - **Large Model**: Bigger architecture (Context=7, Embed=256, Hidden=512)
        - **GELU Variant**: Modern GELU activation function
        
        *Try different variants to see how architecture affects generation!*
        """)
    
    with st.spinner(f"Loading {selected_model_name}..."):
        model_info = load_model(model_path)
    
    if model_info is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    try:
        model_exists = os.path.exists(model_path)
        model_size = os.path.getsize(model_path) if model_exists else None
    except Exception:
        model_exists = False
        model_size = None

    st.sidebar.success("Model loaded!")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Model Architecture")
    
    with st.sidebar.expander("View Model Details", expanded=False):
        st.write(f"**Vocabulary Size:** {model_info['vocab_size']:,} words")
        st.write(f"**Embedding Dimension:** {model_info['embedding_dim']}")
        st.write(f"**Hidden Dimension:** {model_info['hidden_dim']}")
        st.write(f"**Context Length:** {model_info['context_length']} words")
        st.write(f"**Activation Function:** {model_info['activation'].upper()}")
        
        if model_info.get('training_accuracy') != 'N/A':
            st.write(f"**Training Accuracy:** {model_info['training_accuracy']:.2%}")
        if model_info.get('training_loss') != 'N/A':
            st.write(f"**Best Loss:** {model_info['training_loss']:.4f}")
        
        st.markdown("---")
        st.markdown("**Fixed Parameters (from trained model):**")
        st.write(f"- Context Length: {model_info['context_length']}")
        st.write(f"- Embedding Dim: {model_info['embedding_dim']}")
        st.write(f"- Activation: {model_info['activation'].upper()}")
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Generation Controls")
    
    num_to_generate = st.sidebar.slider(
        "Number of words to generate",
        min_value=5,
        max_value=150,
        value=30,
        step=5,
        help="How many words to generate after the seed text"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Sampling Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls randomness: Lower = more deterministic, Higher = more creative"
    )
    
    if temperature < 0.7:
        temp_desc = "**Conservative**: More predictable, safer outputs"
    elif temperature < 1.3:
        temp_desc = "**Balanced**: Good mix of coherence and creativity"
    else:
        temp_desc = "**Creative**: More diverse but potentially less coherent"
    
    st.sidebar.caption(temp_desc)
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("Reproducibility")
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="Set seed for reproducible results"
    )
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("Debug Info"):
        st.write(f"Model path: `{os.path.basename(model_path)}`")
        st.write(f"Exists: {model_exists}")
        if model_size is not None:
            st.write(f"Size: {model_size / 1024 / 1024:.2f} MB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        if is_code:
            default_examples = [
                "struct task_struct *",
                "if ( err )",
                "static int",
                "return"
            ]
            placeholder = "struct task_struct *"
        else:
            default_examples = [
                "to be or not to",
                "what is the meaning of",
                "i think that",
                "the king of"
            ]
            placeholder = "to be or not to"
        
        st.markdown("**Quick Examples (click to use):**")
        example_cols = st.columns(len(default_examples))
        for i, example in enumerate(default_examples):
            if example_cols[i].button(f"`{example}`", key=f"ex_{i}"):
                st.session_state.seed_text = example
        
        st.markdown("---")
        
        seed_text = st.text_area(
            "Enter seed text:",
            value=st.session_state.get('seed_text', placeholder),
            height=120,
            help="Enter a few words to start generation. The model will continue from here.",
            placeholder="Type your seed text here..."
        )
        
        generate_button = st.button("Generate Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Output")
        
        if generate_button:
            if not seed_text.strip():
                st.warning("Please enter some seed text to start generation.")
            else:
                with st.spinner("Generating text..."):
                    try:
                        generated, oov_info = generate_text(
                            model_info,
                            seed_text,
                            num_to_generate,
                            temperature,
                            is_code
                        )
                        
                        st.markdown("**Generated Text:**")
                        st.text_area(
                            "Result",
                            value=generated,
                            height=250,
                            label_visibility="collapsed"
                        )
                        
                        word_count = len(generated.split())
                        seed_word_count = len(seed_text.split())
                        actual_new = oov_info.get('generated_count', word_count - seed_word_count)
                        st.caption(f"Generated {word_count} words total ({actual_new} new words)")
                        
                        with st.expander("Debug Info"):
                            st.write(f"**Seed words:** {seed_word_count}")
                            st.write(f"**Total output words:** {word_count}")
                            st.write(f"**New words generated:** {actual_new}")
                            st.write(f"**Requested:** {num_to_generate} words")
                            st.write(f"**Temperature:** {temperature}")
                            st.write(f"**Model type:** {'Code' if is_code else 'Text'}")
                            
                            if 'debug_tokens' in oov_info:
                                st.markdown("---")
                                st.markdown("**Generation Steps:**")
                                for debug_line in oov_info['debug_tokens'][:10]:
                                    st.text(debug_line)
                        
                        if oov_info['total_oov'] > 0:
                            with st.expander("Out-of-Vocabulary (OOV) Words Detected", expanded=True):
                                st.warning(f"Found {oov_info['total_oov']} word(s) not in the model's vocabulary.")
                                st.write("**OOV words were replaced with `<UNK>` token:**")
                                for word in oov_info['oov_words']:
                                    st.code(word, language=None)
                                st.info("""
                                **Tip:** For better results, try using words that are more common in the training 
                                dataset, or use one of the example prompts above.
                                """)
                        else:
                            st.success("All words found in vocabulary!")
                        
                    except Exception as e:
                        st.error(f"Error during generation: {str(e)}")
                        st.exception(e)
        else:
            st.info("Configure settings in the sidebar and click **Generate Text** to start")


if __name__ == "__main__":
    main()
