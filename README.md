# Metric Differential Privacy for NLP

Given a **Vocabulary** [Dictionary] and a lexical **Embedding** [Array], e.g., ```GloVe``` or ```BERT```, this repository provides a ```Class``` to perturb numeralized tokens. Approximate Nearest Neighbors are calculated using ```Annoy```. Calibration of Multivariate Perturbations are explained in ```Feyisetan et al. (2020)```.

#### All in One

```python
text = "The text you provided contains some random words and symbols that don't form a coherent sentence or question. If you have a specific question or topic you'd like assistance with, please let me know, and I'll be happy to help you."
max_length = 100
from mdp import metricDP
from transformers import BertTokenizer
from transformers import BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocabulary = tokenizer.vocab
model = BertModel.from_pretrained("bert-base-uncased")
embedding = model.embeddings.word_embeddings.weight.cpu().detach().numpy()
mdp = metricDP(vocabulary, embedding, start_from=999)
mdp.build_ann(metric='euclidean', n_trees=50)
ids = tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_length)
priv_ids = mdp.privatize(ids, epsilon=1, special_tokens=[0,100,101,102,103])
priv_text = tokenizer.decode(priv_ids)
print(priv_text)
```


#### Example

We use ``BERT`` from ```HuggingFace``` to provide an example.

1. Extract Vocabulary

```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocabulary = tokenizer.vocab
```
2. Extract Embedding
```
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-uncased")
embedding = model.embeddings.word_embeddings.weight.cpu().detach().numpy()
```
3. Initialize Class
```
from mdp import metricDP
mdp = metricDP(vocabulary, embedding, start_from=999)
mdp.build_ann(metric='euclidean', n_trees=50)
```
To exclude special tokens from the candidate pool, specifiy the position of regular tokens via ```start_from```. In BERT, the first regular token is '!' at index 999. During the privatization step, each token is remaped from its nearest neighbor item to the embedding index. 

4. Numeralize Input
```
text = 'The cat sat on the mat.'
ids = tokenizer.encode(text, truncation=True, padding='max_length', max_length=10)
# [101, 1996, 4937, 2938, 2006, 1996, 13523, 1012, 102, 0]
```
5. Privatize Input
```
mdp.privatize(ids, epsilon=1, special_tokens=[0,100,101,102,103])
#[101, 2601, 2267, 25195, 20139, 6584, 16304, 22754, 102, 0]
```
Perturbations ignore all tokens specified in ```special_tokens```, and ```epsilon``` regulates the privacy guarantees. A smaller epsilon leads to more perturbations and higher privacy guarantees. A higher epsilon leads to less perturbations and lower privacy guarantees.

#### Citation

      @InProceedings{feyisetan2020privacy,
        author="Feyisetan, Oluwaseyi and Balle, Borja and Drake, Thomas and Diethe, Tom",
        title="Privacy-and utility-preserving textual analysis via calibrated multivariate perturbations",
        booktitle="Proceedings of the 13th International Conference on Web Search and Data Mining",
        year="2020",
      }
