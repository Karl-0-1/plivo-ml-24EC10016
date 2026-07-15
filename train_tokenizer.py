import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def main():
    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE())
    
    # Use ByteLevel pre-tokenizer to ensure lossless arbitrary byte encoding
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Train it!
    # We choose a vocab size of 4096 to balance sequence length compression with parameter constraints.
    trainer = BpeTrainer(vocab_size=4096, special_tokens=["<|endoftext|>"])
    
    # Train on the corpus
    corpus_path = "../data/train_corpus.txt"
    tokenizer.train([corpus_path], trainer)
    
    # Setup decoder
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tokenizer.decoder = ByteLevelDecoder()
    
    # Save the tokenizer
    tokenizer.save("bpe_tokenizer.json")
    print("Tokenizer trained and saved to bpe_tokenizer.json")

if __name__ == "__main__":
    main()
