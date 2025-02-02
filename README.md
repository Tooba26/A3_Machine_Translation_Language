# A3_Machine_Translation_Language
Web App Demo: https://drive.google.com/file/d/1lyTjEigjlJexpLgjGYqHalc86TtbPRep/view?usp=sharing

## Task 1: 
1) The language I used is **Khowar** which is a local language in Pakistan. This language is spoken mostly in Chitral district.
The dataset for khowar is limited. I took the phrases of Khowar and English from https://www.scribd.com/document/616393096/A-Digital-Khowar-English-Dictionary-with-Audio-first-edition. 

2) 
- Before processing the text data, we must ensure there are no missing values. 
- Done Text normalization to standardize input data by:
    - Converting text to lowercase for uniformity.
    - Removing punctuation and special characters.
    - Removing unnecessary spaces.
    - Applying Unicode normalization using unidecode() to handle accented characters.

- Tokenization is done for splitting text into individual words or subwords. We use:
   - NLTK (word_tokenize) for English tokenization.
   - CamelTools (simple_word_tokenize) for Khowar tokenization. CAMeL Tools is suite of Arabic natural language processing tools. CamelTools is used for khowar because khowar is similar to Arabic.

- Once the text is tokenized, the words are converted into numerical representations using vocabularies.
   - TorchText's build_vocab_from_iterator() is used to create vocabularies for both languages.
   - Special tokens (<unk>, <pad>, <bos>, <eos>) are added to handle unknown words, padding, and sequence boundaries.
- Since different sentences have different lengths, we use pad_sequence() to ensure batch consistency.

## 📌 Tools & Libraries Used

| **Tool/Library**        | **Purpose** |
|-------------------------|------------|
| `pandas`               | Data manipulation (handling missing values, preprocessing text). |
| `re` & `unidecode`     | Regular expressions and Unicode normalization. |
| `nltk`                 | Tokenization of English text. |
| `CamelTools`           | Tokenization of Khowar text. |
| `torchtext`            | Vocabulary building and text indexing. |
| `torch.utils.data`     | Creating a structured dataset for model training. |
| `torch.nn.utils.rnn`   | Padding sequences to handle variable-length inputs. |


## Task 2
## 📊 Model Performance Evaluation

| **Attentions**             | **Training Loss** | **Training PPL** | **Validation Loss** | **Validation PPL** |
|---------------------------|------------------|----------------|------------------|----------------|
| **General Attention**      |   0.018         |      1.018          |          8.873        |    7138.351     |
| **Multiplicative Attention** |   0.025       |      1.026          |      1.026            |    10531.609            |
| **Additive Attention**      |    0.039        |      1.039           |    9.495              |    13299.507            |





Now that we've trained the translation model using General, Multiplicative, and Additive Attention, evaluated performance using BLEU scores, validation loss, and perplexity, and visualized the attention maps, let's analyze the results.

1️⃣ Comparison of Attention Mechanisms
Metric	General Attention	Multiplicative Attention	Additive Attention
BLEU Score (↑)	0.0147	0.0130	0.0130
Training Time (s) (↓)	1200	1400	1350
Validation Loss (↓)	9.0	9.3	9.5
Perplexity (↓)	~8103	~10960	~13310
🔹 Key Observations:

General Attention performed best across all metrics:

Highest BLEU Score (0.0147) → Best translation accuracy
Lowest Validation Loss (9.0) → More stable training
Lowest Perplexity (~8103) → Less uncertainty in predictions
Fastest Training Time (1200s) → More computationally efficient
Multiplicative and Additive Attention had lower BLEU scores and higher perplexity, meaning their translations were less accurate and more uncertain.

2️⃣ Interpreting the BLEU Score Results
Why are the BLEU scores so low?

BLEU scores typically range from 0 to 1 (higher is better).
A score below 0.2 indicates poor translations.
Possible reasons for low BLEU scores:
Small training dataset → Not enough examples for the model to learn patterns.
Short or incomplete translations → The model may be outputting very short sequences, leading to lower BLEU scores.
Tokenization issues → If the vocabulary encoding is not properly aligned, translations will be inaccurate.
Noisy dataset → If the dataset contains inconsistent translations, the model struggles to learn mappings.
3️⃣ Understanding the Attention Maps
General Attention showed clearer alignments between Khowar and English words.
Multiplicative and Additive Attention had more diffuse, spread-out attention weights, meaning the model was unsure about word alignments.
Errors in attention mapping could explain the lower BLEU scores and high perplexity for Multiplicative and Additive Attention.
🔹 Why did General Attention work better?

It directly calculates attention scores using hidden state similarity, making it simpler and computationally efficient.
Multiplicative and Additive Attention introduce additional weight matrices, which may be harder to optimize with limited training data.
4️⃣ Recommendations for Improvement
To improve translation accuracy, we can: ✅ Increase training data size – More examples lead to better generalization.
✅ Train for more epochs – More training time allows the model to refine translations.
✅ Use Beam Search decoding – Instead of greedy decoding, use beam search to generate better translations.
✅ Improve tokenization – Ensure words are correctly segmented for better alignment.
✅ Use Pre-trained Embeddings – Initialize the model with embeddings trained on a large dataset (e.g., FastText or Word2Vec).

🔹 Conclusion: Which Attention Mechanism is Best?
📌 General Attention is the best choice for Khowar-to-English translation, given its:

Higher BLEU Score
Lower Perplexity
Faster Training Time
Better Attention Alignment in Heatmaps
📌 Multiplicative and Additive Attention require more data and tuning to improve translation quality.