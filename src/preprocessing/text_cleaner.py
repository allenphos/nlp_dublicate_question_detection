import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class TextCleaner:
    """
    Handles text preprocessing:
    - Lowercasing
    - Removing non-letter chars
    - Reducing repeated chars
    - Tokenizing
    - Removing stopwords
    - Stemming
    """
    def __init__(self, language: str = "english"):
        # Load stopwords for the specified language (e.g. English)
        self.stopwords = set(stopwords.words(language))
        # Initialize a stemmer for the specified language
        self.stemmer = SnowballStemmer(language)

    def reduce_lengthening(self, text):
        # Replace repeated characters (e.g. "coooool" -> "cool")
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def __call__(self, text: str) -> str:
        # Remove all non-letter characters (replace with space)
        text = re.sub(r"[^a-zA-Z]", " ", str(text))
        # Reduce repeated characters (see above)
        text = self.reduce_lengthening(text)
        # Split text into tokens (words)
        tokens = word_tokenize(text)
        # Lowercase and filter out stopwords
        tokens = [t.lower() for t in tokens if t.lower() not in self.stopwords]
        # Apply stemming to each token
        stems = [self.stemmer.stem(token) for token in tokens]
        # Join tokens back into a string
        return " ".join(stems)
