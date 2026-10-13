"""Aspect-based sentiment (Member 2 extension).

A review often praises the food and complains about the wait. Overall polarity hides that, so
each review is split into sentences, sentences are matched to aspects by keyword, and the
trained TF-IDF sentiment pipeline scores each matched sentence. Aggregating P(positive) per
aspect gives, for a business or an area, *what* customers like or dislike.
The keyword lists are a transparent, editable lexicon (not learned); a sentence can match
several aspects.
"""
import re

import numpy as np
import pandas as pd

from .competitor_analysis import normalize_text

ASPECTS = {
    'Food & taste': ['food', 'taste', 'tasty', 'flavor', 'flavour', 'delicious', 'bland', 'dish', 'dishes', 'meal', 'portion',
                     'portions', 'fresh', 'cooked', 'menu', 'pizza', 'burger', 'sandwich', 'chicken', 'steak', 'coffee', 'sauce'],
    'Service & staff': ['service', 'staff', 'server', 'servers', 'waiter', 'waitress', 'bartender', 'rude', 'friendly',
                        'attentive', 'manager', 'host', 'hostess', 'owner', 'polite'],
    'Wait time': ['wait', 'waited', 'waiting', 'minutes', 'hour', 'slow', 'quick', 'quickly', 'fast', 'line', 'forever'],
    'Price & value': ['price', 'prices', 'priced', 'expensive', 'cheap', 'overpriced', 'value', 'worth', 'cost', 'bill', 'tip'],
    'Ambience': ['ambience', 'ambiance', 'atmosphere', 'decor', 'music', 'loud', 'noisy', 'cozy', 'vibe', 'seating', 'view'],
    'Cleanliness': ['clean', 'dirty', 'filthy', 'bathroom', 'restroom', 'sticky', 'smell', 'hygiene', 'roach', 'gross'],
    'Delivery & orders': ['delivery', 'delivered', 'order', 'ordered', 'pickup', 'takeout', 'grubhub', 'ubereats',
                          'doordash', 'phone', 'called', 'wrong', 'missing'],
}
_SENTENCE = re.compile(r'(?<=[.!?])\s+|\n+')
_PATTERNS = {aspect: re.compile(r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b') for aspect, words in ASPECTS.items()}


def aspect_sentences(texts, ids=None):
    """One row per (review, sentence, aspect) match with the cleaned sentence text."""
    ids = range(len(texts)) if ids is None else ids
    rows = []
    for review_id, text in zip(ids, texts):
        for sentence in _SENTENCE.split(str(text)):
            clean = normalize_text([sentence]).iloc[0]
            if len(clean.split()) < 3:
                continue
            for aspect, pattern in _PATTERNS.items():
                if pattern.search(clean):
                    rows.append({'review_id': review_id, 'aspect': aspect, 'sentence': clean})
    return pd.DataFrame(rows, columns=['review_id', 'aspect', 'sentence'])


def score_aspects(texts, sentiment_model, ids=None, threshold=0.5):
    """Sentence-level aspect matches with P(positive) from the saved sentiment pipeline."""
    matches = aspect_sentences(texts, ids)
    if matches.empty:
        return matches.assign(prob_positive=pd.Series(dtype=float), positive=pd.Series(dtype=int))
    matches['prob_positive'] = sentiment_model.predict_proba(matches['sentence'])[:, 1]
    matches['positive'] = (matches['prob_positive'] >= threshold).astype(int)
    return matches


def summarize_aspects(scored, group_col=None, min_mentions=1):
    """Mentions, share of positive mentions and mean P(positive) per aspect (optionally per group)."""
    keys = ['aspect'] if group_col is None else [group_col, 'aspect']
    out = (scored.groupby(keys).agg(mentions=('prob_positive', 'size'), positive_share=('positive', 'mean'),
                                    mean_prob_positive=('prob_positive', 'mean')).reset_index())
    out['positive_share'] *= 100
    return out[out['mentions'] >= min_mentions].reset_index(drop=True)
