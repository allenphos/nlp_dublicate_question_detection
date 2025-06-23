import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, log_loss, classification_report, confusion_matrix

class LSTMTrainer:
    """
    Object-oriented trainer for an LSTM-based sequence model for duplicate question detection.
    Handles tokenization, padding, splitting, training, and evaluation.
    """

    def __init__(self, max_num_words=15000, max_seq_len=60):
        """
        Initialize the LSTMTrainer.
        
        Args:
            max_num_words (int): Maximum number of words in the vocabulary.
            max_seq_len (int): Maximum length of input sequences (after padding).
        """
        self.max_num_words = max_num_words
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
        self.model = None

    def prepare_sequences(self, df, col1='question1', col2='question2'):
        """
        Combines two columns into one string per row and prepares sequences for LSTM.
        
        Args:
            df (pd.DataFrame): DataFrame with question pairs.
            col1 (str): Name of the first question column.
            col2 (str): Name of the second question column.
        
        Returns:
            X (np.array): Padded integer sequences.
            y (np.array): Target binary labels.
        """
        # Combine question1 and question2 with a separator token
        df['combined'] = df[col1].astype(str) + ' [SEP] ' + df[col2].astype(str)
        # Fit tokenizer only on train data!
        self.tokenizer.fit_on_texts(df['combined'])
        # Convert combined text to integer sequences
        sequences = self.tokenizer.texts_to_sequences(df['combined'])
        # Pad/truncate all sequences to the same length
        X = pad_sequences(sequences, maxlen=self.max_seq_len, padding='post', truncating='post')
        # Extract labels
        y = df['is_duplicate'].values
        return X, y

    def train_val_test_split(self, X, y, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            X (np.array): Feature matrix.
            y (np.array): Labels.
            val_size (float): Fraction of temp set for validation.
            test_size (float): Fraction of all data for test set.
            random_state (int): Seed for reproducibility.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Arrays for training.
        """
        # First, separate out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        # Then split the rest into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        """
        Build a Keras LSTM model for sequence classification.
        """
        model = Sequential([
            Embedding(input_dim=self.max_num_words, output_dim=128, input_length=self.max_seq_len),
            Bidirectional(LSTM(64)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=128):
        """
        Train the LSTM model.
        
        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.
        """
        # Use early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set. Prints all standard metrics.
        
        Args:
            X_test, y_test: Test features and labels.
        
        Returns:
            dict with f1 and logloss
        """
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        print("F1-score:", f1_score(y_test, y_pred))
        print("Log loss:", log_loss(y_test, y_pred_proba))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification report:\n", classification_report(y_test, y_pred))
        return {
            "f1": f1_score(y_test, y_pred),
            "logloss": log_loss(y_test, y_pred_proba)
        }
