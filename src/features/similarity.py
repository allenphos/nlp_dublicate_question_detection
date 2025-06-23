import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from transformers import BertTokenizer, BertModel
import torch
from typing import Dict, List


class TextEmbedder:
    """Batch BERT [CLS] embedding extractor with .npy export."""
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def get_cls_embeddings(self, texts, max_length=64, batch_size=64, save_path=None):
        """Return {text: [CLS] embedding} for unique texts, optionally save to .npy."""
        unique_texts = list(set(texts))
        embeddings = []
        text_order = []
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="BERT embeddings"):
            batch = unique_texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model(**inputs)
            batch_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)
            text_order.extend(batch)
            del batch_emb, output, inputs
            gc.collect()
        # Stack and create mapping
        emb_mat = np.vstack(embeddings)
        if save_path is not None:
            np.save(save_path, emb_mat)
            print(f"Saved to {save_path}, shape: {emb_mat.shape}")
        return dict(zip(text_order, emb_mat))

class TextSimilarity:
    """Compute similarity metrics between question pairs using given embeddings."""
    def __init__(self, embeddings_dict: Dict[str, np.ndarray]):
        self.embeddings_dict = embeddings_dict

    def _get_pair_embeddings(self, q1_list: List[str], q2_list: List[str]):
        q1_embs = np.stack([self.embeddings_dict[q] for q in q1_list])
        q2_embs = np.stack([self.embeddings_dict[q] for q in q2_list])
        return q1_embs, q2_embs

    def cosine(self, q1_list: List[str], q2_list: List[str]) -> np.ndarray:
        q1_embs, q2_embs = self._get_pair_embeddings(q1_list, q2_list)
        # Cosine similarity for each pair
        return np.array([
            cosine_similarity(q1_embs[i].reshape(1, -1), q2_embs[i].reshape(1, -1))[0][0]
            for i in range(len(q1_list))
        ])

    def manhattan(self, q1_list: List[str], q2_list: List[str]) -> np.ndarray:
        q1_embs, q2_embs = self._get_pair_embeddings(q1_list, q2_list)
        return np.array([
            manhattan_distances(q1_embs[i].reshape(1, -1), q2_embs[i].reshape(1, -1))[0][0]
            for i in range(len(q1_list))
        ])

    def euclidean(self, q1_list: List[str], q2_list: List[str]) -> np.ndarray:
        q1_embs, q2_embs = self._get_pair_embeddings(q1_list, q2_list)
        return np.array([
            euclidean_distances(q1_embs[i].reshape(1, -1), q2_embs[i].reshape(1, -1))[0][0]
            for i in range(len(q1_list))
        ])
        
def compute_and_save_bert_features(df, embedder, out_prefix, batch_size=64):
    all_questions = pd.concat([df['question1'], df['question2']]).astype(str).unique().tolist()
    print(f"Number of unique questions: {len(all_questions)}")
    emb_path = f"{out_prefix}_embeddings.npy"
    q2emb = embedder.get_cls_embeddings(all_questions, batch_size=batch_size, save_path=emb_path)

    q1_embs = np.stack([q2emb[str(q)] for q in df['question1']])
    q2_embs = np.stack([q2emb[str(q)] for q in df['question2']])
    np.save(f"{out_prefix}_q1.npy", q1_embs)
    np.save(f"{out_prefix}_q2.npy", q2_embs)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = [cosine_similarity(q1_embs[i].reshape(1,-1), q2_embs[i].reshape(1,-1))[0,0] for i in range(len(df))]
    df['bert_cosine_similarity'] = sim

    # Save dataframe with similarity
    df.to_csv(f"{out_prefix}_with_bert_sim.csv", index=False)
    print(f"Saved {out_prefix}_with_bert_sim.csv and .npy embeddings.")
    del q1_embs, q2_embs, sim, q2emb
    gc.collect()
