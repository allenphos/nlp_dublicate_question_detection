# Duplicate Question Detection with NLP

A machine learning project to identify semantically duplicate questions using the Quora dataset. Combines classical ML models and fine-tuned transformer-based architectures.

---

## Project Goal

Classify whether two questions are duplicates by analyzing their semantic similarity. The project explores traditional machine learning, deep learning, and transfer learning methods for natural language understanding.

---

## 📑 Table of Contents

- [🎯 Project Goal](#project-goal)
- [📁 Project Structure](#project-structure)
- [🧰 Tools & Technologies](#tools--technologies)
- [🧪 Methodology](#methodology)
- [📊 Models & Performance](#models--performance)
- [✅ Evaluation](#evaluation)
- [📦 Data Files](#project-structure)
- [📜 Requirements](#requirements)
- [🔚 Conclusion](#conclusion)
- [📈 Next Steps](#next-steps)
- [👩‍💻 Author](#author)

---

## Tools & Technologies

- **Python** — core programming language used for all scripts and modeling
- **Jupyter Notebooks** — for interactive development, prototyping, and documentation
- **Scikit-learn** — classical ML models, metrics, and pipelines
- **Pandas / NumPy** — data manipulation and numerical operations
- **TensorFlow / PyTorch** — deep learning frameworks for custom model training
- **Hugging Face Transformers** — used for fine-tuning and deploying DistilBERT (LLM)
- **DistilBERT (via Transformers)** — lightweight transformer model fine-tuned for semantic similarity
- **Large Language Models (LLMs)** — transfer learning from pre-trained transformer models
- **Joblib** — efficient serialization of trained models for later reuse
- Project functionality was modularized using Object-Oriented Programming (OOP) principles to enhance reusability, scalability, and maintainability of preprocessing, feature engineering, and model training components.

---

## Project Structure

<pre lang="markdown"><code>## 📁 Project Structure ``` 
  
├── data/
│ ├── processed/
│ │ ├── *.zip, *.csv ← Cleaned & augmented datasets
│ ├── raw/ ← (*.zip raw Quora files)
│
├── notebooks/
│ ├── 01_eda.ipynb ← Exploratory Data Analysis
│ ├── 02_preprocessing_and_feature_eng.ipynb ← Cleaning & feature generation
│ ├── 03_modeling_and_evaluation.ipynb ← Baseline models: TF-IDF, LogReg, RF
│ ├── 04_augmentation.ipynb ← Augmented paraphrases
│ ├── 05_modeling_after_augm.ipynb ← Models after data augmentation
│ ├── 06_lstm_and_bert_fine-tuning.ipynb ← LSTM and fine-tuned DistilBERT
│ ├── 07_evaluation.ipynb ← Final performance & confusion matrices
│
├── reports/
│ └── metrics_and_confusions.json ← All evaluation scores and confusion matrices
│
├── src/
│ ├── analysis/ ← Custom metrics, visualization
│ ├── augmentation/ ← Data augmentation scripts
│ ├── features/ ← Feature engineering modules
│ ├── models/ ← Model training and saving
│ ├── preprocessing/ ← Cleaning, tokenization
│
├── .gitignore
├── .gitattributes
├── requirements.txt
├── README.md ``` </code></pre>

---

## Models & Performance

| **Model**                                  | **Accuracy** | **F1 Score** | **Log Loss** | **Notes**                                                |
|-------------------------------------------|--------------|--------------|--------------|-----------------------------------------------------------|
| BERT (test set)                            | 0.90         | 0.87         | 0.33         | Best real-world performance, strong generalization        |
| BERT (train set)                           | 0.97         | 0.96         | 0.10         | Extremely high accuracy, but overfitted                   |
| LSTM (sample)                              | 0.78         | 0.68         | 0.46         | Lightweight but weaker, lower recall                     |
| LogReg (TF-IDF+BERT)                       | 0.75         | 0.70         | 0.49         | Strong baseline with interpretable weights               |
| RandomForest (TF-IDF+BERT, before augm)    | 0.82         | 0.74         | 0.42         | Best classic ML model                                    |
| XGBoost (TF-IDF+BERT)                      | 0.73         | 0.69         | 0.50         | Good recall but high false positives                     |
| Ensemble (TF-IDF+BERT)                     | 0.79         | 0.74         | 0.44         | Improved generalization via ensembling                   |
| RandomForest (TF-IDF+BERT, after augm)     | 0.79         | 0.70         | 0.50         | Slight performance drop from noisy augmentation          |

---

## Methodology

- Exploratory Data Analysis (EDA)
- Text preprocessing and feature engineering
- TF-IDF and BERT embeddings
- Data augmentation and balancing
- Classical ML: Logistic Regression, Random Forest, XGBoost
- Deep learning: LSTM with embeddings
- Transformer fine-tuning: DistilBERT
- Model evaluation: accuracy, F1-score, log loss, confusion matrices

---

## Evaluation

- Metrics: Accuracy, F1, Log Loss
- Tools: `classification_report`, `confusion_matrix`, `joblib`, `numpy`
- Final performance summary saved in `metrics_and_confusions.json`

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```
---

## Conclusion

This project demonstrated the effectiveness of combining classical machine learning and modern transformer models for detecting semantically similar questions. Fine-tuned DistilBERT achieved the best results, confirming the power of transfer learning in NLP tasks.

### Key Takeaways:

- **BERT-based models** significantly outperform traditional models in both precision and recall.
- **Classical ML models** remain strong baselines, especially with TF-IDF and engineered BERT features.
- **Augmentation** helps expand training data but may introduce noise if not carefully managed.
- **Model evaluation** with confusion matrices and log loss provided a deep look into performance beyond accuracy.

---
## Next Steps:

- 📌 Deploy the best-performing BERT model via a simple API (Streamlit).
- 🧪 Explore other transformer architectures (e.g., RoBERTa, DeBERTa) for further performance gains.
- 📈 Conduct error analysis to identify recurring patterns in false positives/negatives.
- 📦 Package the pipeline into a modular, reusable framework for future NLP tasks.
- ⚙️ Implement regularization techniques (such as Dropout, Early Stopping, or L2) to reduce overfitting and improve model generalization.

---

## Author

Anastasia Alyoshkina
🔗 [LinkedIn]([https://www.linkedin.com/in/your-link](https://www.linkedin.com/in/anastasiia-alyoshkina-68ba5929a/)) | 📬 anastasia.alshkn@gmail.com

---

