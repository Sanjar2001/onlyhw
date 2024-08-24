import pandas as pd
from fastai.text.all import *

# Загрузка датасета
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')

dls = TextDataLoaders.from_df(df, text_col='text', label_col='label', is_lm=True)
dls.show_batch(max_n=5)

learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=[accuracy, Perplexity()]).to_fp16()

learn.fit_one_cycle(1, 1e-2)
learn.save('stage-1')


learn = learn.load('stage-1')
learn.fit_one_cycle(3, 1e-3)
learn.save('stage-2')

learn.save_encoder('fine_tuned_encoder')

TEXT = "The movie was"
n_words = 40
preds = learn.predict(TEXT, n_words, temperature=0.75)
print(preds)

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn = learn.load_encoder('fine_tuned_encoder')
learn.fit_one_cycle(1, 1e-2)
