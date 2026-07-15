import os
from tokenizers import Tokenizer as HFTokenizer

def main():
    tokenizer = HFTokenizer.from_file("bpe_tokenizer.json")
    test_strings = [
        "Hello world!",
        "नमस्ते दुनिया",
        "A mix of English and हिंदी",
        "Some random bytes \x80\x81\xff",
    ]
    for text in test_strings:
        encoded = tokenizer.encode(text).ids
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        assert decoded == text, f"Failed on {text}: decoded to {decoded}"
    
    print("Tokenizer is lossless!")

if __name__ == "__main__":
    main()
