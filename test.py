import time
begin_time = time.perf_counter()

from mdp import metricDP
from transformers import BertTokenizer, BertModel

# for supressing warnings
from transformers import logging
logging.set_verbosity_error()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

text = "Dear Dr. Costa, As promised, we have worked on the revisions and are pleased to present you with the final version of our paper. Please find attached a copy of it, along with a summary of changes. We kindly request your valuable input and feedback on the final version. Should you have any comments or suggestions, we would greatly appreciate hearing them. Regards, Zhifeng"

max_length = 100
epsilon = 200

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocabulary = tokenizer.vocab
model = BertModel.from_pretrained("bert-base-uncased")
embedding = model.embeddings.word_embeddings.weight.cpu().detach().numpy()

mdp = metricDP(vocabulary, embedding, start_from=999)
mdp.build_ann(metric='euclidean', n_trees=50)
ids = tokenizer.encode(
    text,
    truncation=True,
    padding='max_length',
    max_length=max_length,
)

priv_ids = mdp.privatize(ids, epsilon=epsilon, special_tokens=[0,100,101,102,103])
priv_text = tokenizer.decode(
    priv_ids
)

end_time = time.perf_counter()
duration = round(end_time - begin_time, 2)
print(f'Done in {duration}s. Original text:')
print(text)
print('\n\nNew privatized text:')
print(priv_text)