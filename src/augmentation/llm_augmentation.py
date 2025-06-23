import time
import pandas as pd
from sentence_transformers import util

def generate_augmented_pairs(themes, n_pairs, similarity_threshold, prompt, llm, embed_model, delay=1.5):
    """
    Generate paraphrased duplicate question pairs using LLM and filter by semantic similarity.

    Args:
        themes (list of str): List of question topics.
        n_pairs (int): Number of pairs to generate.
        similarity_threshold (float): Minimum cosine similarity to accept a pair as duplicate.
        prompt: LangChain prompt template.
        llm: Initialized LLM model (e.g., ChatOpenAI).
        embed_model: SentenceTransformer model for semantic similarity.
        delay (float): Delay between API calls to avoid rate limits.

    Returns:
        pd.DataFrame: DataFrame with columns ['question1', 'question2', 'is_duplicate']
    """
    pairs = []
    existing_pairs = set()
    for i in range(n_pairs):
        try:
            theme = themes[i % len(themes)]
            filled_prompt = prompt.format(theme=theme)
            output = llm.invoke(filled_prompt)
            lines = output.content.strip().split("\n")
            if len(lines) < 2:
                print(f"✗ {i+1}: Output parsing failed, skipping.")
                continue
            q1 = lines[0].replace("Q1:", "").strip()
            q2 = lines[1].replace("Q2:", "").strip()
            # Skip duplicates
            key = (q1, q2)
            rev_key = (q2, q1)
            if key in existing_pairs or rev_key in existing_pairs:
                print(f"✗ {i+1}: Duplicate detected, skipping.")
                continue
            emb1 = embed_model.encode(q1, convert_to_tensor=True)
            emb2 = embed_model.encode(q2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            if similarity > similarity_threshold:
                pairs.append({"question1": q1, "question2": q2, "is_duplicate": 1})
                existing_pairs.add(key)
                print(f"✓ {i+1}: Passed filter ({similarity:.2f})")
            else:
                print(f"✗ {i+1}: Failed filter ({similarity:.2f})")
            time.sleep(delay)
        except Exception as e:
            print(f"Error at {i+1}:", e)
            continue
    return pd.DataFrame(pairs)
