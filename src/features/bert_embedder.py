from transformers import BertTokenizer, BertModel
import torch
from typing import List, Dict

class BertEmbedder:
    """
    Wrapper for extracting BERT embeddings using HuggingFace Transformers.
    """
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Disable dropout for evaluation

    def get_embeddings(self, text: str, max_length: int = 64) -> Dict:
        """
        Return last hidden states and CLS token embedding for a given text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0).cpu()
        cls_embedding = last_hidden[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())
        return {
            "tokens": tokens,
            "last_hidden_states": last_hidden,
            "cls_embedding": cls_embedding
        }

    def get_batch_cls(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        """
        Get [CLS] embeddings for a list of texts.
        """
        inputs = self.tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CLS = always first token
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # shape: (batch, 768)
        return cls_embeddings

    def show_embedding_info(self, text: str):
        """
        Print embedding info for a sample sentence.
        """
        result = self.get_embeddings(text)
        print(f"Number of tokens: {len(result['tokens'])}")
        print("Tokens and embedding shapes (first 5):")
        for token, emb in zip(result['tokens'][:5], result['last_hidden_states'][:5]):
            print(f"  {token:>10} | {emb.shape} | {emb[:5]}...")
        print("CLS embedding (first 10 values):", result['cls_embedding'][:10].numpy())

