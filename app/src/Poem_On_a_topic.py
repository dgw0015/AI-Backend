#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load data
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython

sns.set(context='talk', style='ticks')
get_ipython().run_line_magic('matplotlib', 'inline')

# In[50]:


# Read text file into dataframe and split songs into rows
df = pd.DataFrame({'lyrics': io.open('poemdata.txt', 'r', encoding='ascii', errors='ignore').read().split('\n\n')})

# In[51]:


# Derive text related metrics (number of characters, words, lines, unique words) and lexical density for each poem
# characters, words, lines
df['#characters'] = df.lyrics.str.len()
df['#words'] = df.lyrics.str.split().str.len()
df['#lines'] = df.lyrics.str.split('\n').str.len()
df['#uniq_words'] = df.lyrics.apply(lambda x: len(set(x.split())))
df['lexical_density'] = df['#uniq_words'] / df['#words']

# In[52]:


# Now that we have text metrics, a quick histogram spread on all metrics.
df.hist(sharey=True, layout=(2, 3), figsize=(15, 8));

# In[53]:


# Word length distribution
pd.Series(len(x) for x in ' '.join(df.lyrics).split()).value_counts().sort_index().plot(kind='bar', figsize=(12, 3))

# In[54]:


# top words
pd.Series(' '.join(df.lyrics).lower().split()).value_counts()[:20][::-1].plot(kind='barh')

# In[55]:


# top long words
pd.Series([w for w in ' '.join(df.lyrics).lower().split() if len(w) > 7]).value_counts()[:20][::-1].plot(kind='barh')

# In[56]:


from nltk import ngrams


# In[57]:


def get_ngrams_from_series(series, n=2):
    # using nltk.ngrams
    lines = ' '.join(series).lower().split('\n')
    lgrams = [ngrams(l.split(), n) for l in lines]
    grams = [[' '.join(g) for g in list(lg)] for lg in lgrams]
    return [item for sublist in grams for item in sublist]


# In[58]:


# Top bi-grams
pd.Series(get_ngrams_from_series(df.lyrics, 2)).value_counts()[:20][::-1].plot(kind='barh')

# In[59]:


# Top tri-grams
pd.Series(get_ngrams_from_series(df.lyrics, 3)).value_counts()[:20][::-1].plot(kind='barh')

# In[60]:


# Top four-grams
pd.Series(get_ngrams_from_series(df.lyrics, 4)).value_counts()[:20][::-1].plot(kind='barh')

# In[67]:


# sentiment
import nltk
from nltk import sentiment

nltk.download('vader_lexicon')

# In[68]:


senti_analyze = sentiment.vader.SentimentIntensityAnalyzer()
senti_analyze.polarity_scores(df.lyrics[0])

# In[69]:


df['sentiment_score'] = pd.DataFrame(df.lyrics.apply(senti_analyze.polarity_scores).tolist())['compound']
df['sentiment'] = pd.cut(df['sentiment_score'], [-np.inf, -0.35, 0.35, np.inf],
                         labels=['negative', 'neutral', 'positive'])

# In[70]:


# generating via markov chain
# Machine generated lyrics using Markov
import re
import random
from collections import defaultdict


class GeneratePoetry:
    def __init__(self, corpus='', order=2, length=8):
        self.order = order
        self.length = length
        self.words = re.findall("[a-z']+", corpus.lower())
        self.states = defaultdict(list)

        for i in range(len(self.words) - self.order):
            self.states[tuple(self.words[i:i + self.order])].append(self.words[i + order])

    def gen_sentence(self, length=8, startswith=None):
        terms = None
        if startswith:
            start_seed = [x for x in self.states.keys() if startswith in x]
            if start_seed:
                terms = list(start_seed[0])
        if terms is None:
            start_seed = random.randint(0, len(self.words) - self.order)
            terms = self.words[start_seed:start_seed + self.order]

        for _ in range(length):
            terms.append(random.choice(self.states[tuple(terms[-self.order:])]))

        return ' '.join(terms)

    def gen_song(self, lines=10, length=8, length_range=None, startswith=None):
        song = []
        if startswith:
            song.append(self.gen_sentence(length=length, startswith=startswith))
            lines -= 1
        for _ in range(lines):
            sent_len = random.randint(*length_range) if length_range else length
            song.append(self.gen_sentence(length=sent_len))
        return '\n'.join(song)


# In[71]:


poetrygeneration = GeneratePoetry(corpus=' '.join(df.lyrics))

# In[73]:


poetrygeneration.gen_song(lines=10, length_range=[5, 10], startswith='bird')

# In[ ]:
