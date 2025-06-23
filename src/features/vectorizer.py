from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorizer:
    """
    Wrapper for BoW or TF-IDF vectorization.
    """
    def __init__(self, max_features=9000, mode="tfidf"):
        # Select which type: "tfidf" or "bow"
        if mode == "bow":
            self.vectorizer = CountVectorizer(max_features=max_features)
        elif mode == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            raise ValueError("Unknown mode. Use 'tfidf' or 'bow'.")

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
