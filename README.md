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


## Task 3
Now that we've trained the translation model using General, Multiplicative, and Additive Attention, evaluated performance using BLEU scores, validation loss, and perplexity, and visualized the attention maps, let's analyze the results.

1️⃣ Comparison of Attention Mechanisms
Training time was same for all. The training was done using Epochs = 10. (Initially i did 10 epochs. Later when i tried for more epochs then  there was some memory error in puffer)

General Attention performed best across all metrics:

Highest BLEU Score (0.0147) → Highest translation accuracy among all
Lowest Validation Loss (9.0) → More stable training
Lowest Perplexity (~8103) → Less uncertainty in predictions
Fastest Training Time (1200s) → More computationally efficient

Multiplicative and Additive Attention had lower BLEU scores and higher perplexity, meaning their translations were less accurate and more uncertain.

- General Attention showed clearer alignments between Khowar and English words.
Multiplicative and Additive Attention had more diffuse, spread-out attention weights, meaning the model was unsure about word alignments.
Errors in attention mapping could explain the lower BLEU scores and high perplexity for Multiplicative and Additive Attention.
It directly calculates attention scores using hidden state similarity, making it simpler and computationally efficient.
Multiplicative and Additive Attention introduce additional weight matrices, which may be harder to optimize with limited training data.

2) 
![Validation Loss](/images/Validation_loss.png)
![Training Loss](/images/train_loss.png)

3) 
![Attention General](/images/Attention_general.png)
![Attention Multiplicative](/images/Attention_multi.png)
![Attention Additive](/images/attention_additive.png)

4) 
From the results it can be seen that General attention is providing some better perfomrance than Multiplicative and Additive. 

## Task 4
For the web app i used FastAPI for the backend and ReactJS for frontend. For some styling I used Inline CSS and Material UI.
The web application interfaces with the machine translation model through a React.js frontend and a FastAPI backend. When a user enters text in English, the frontend sends a POST request to the FastAPI server at the /translate endpoint using axios. The backend processes the request by normalizing the text (converting to lowercase, removing punctuation, and handling Unicode characters) and tokenizing it using NLTK for English and CamelTools for Khowar. The tokenized text is then converted into numerical input using a predefined vocabulary and fed into a PyTorch-based Transformer model. The trained models (General, Multiplicative, and Additive Attention) generate translations, which are decoded back into human-readable text and returned as a JSON response to the frontend. The frontend then dynamically updates the UI to display the translation results in a structured Material-UI table, listing translations from all three models. A loading effect is shown while the translation is being processed. 
