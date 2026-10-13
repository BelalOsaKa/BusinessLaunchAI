# Competitor review analysis around a proposed location (Member 2, BusinessLaunch AI).
# Used by the notebook and by streamlit_nlp.py. It works on the files written by the notebook.
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
MIN_REVIEWS_FOR_EVIDENCE = 100
COMPETITOR_COLUMNS = ['name', 'categories', 'is_open', 'distance_km', 'reviews_analyzed', 'avg_rating',
                      'positive_pct', 'neutral_pct', 'negative_pct', 'top_complaint_topic']


def normalize_text(texts):
    # Lowercase, remove URLs and punctuation, turn contractions like "don't" into "do not".
    s = pd.Series(texts).astype(str).str.lower()
    s = s.str.replace('\u2019', "'", regex=False)
    s = s.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)
    s = s.str.replace("won't", 'will not', regex=False).str.replace("can't", 'can not', regex=False)
    s = s.str.replace(r"n't\b", ' not', regex=True)
    s = s.str.replace(r"'(?:s|m|re|ve|ll|d)\b", '', regex=True)
    s = s.str.replace(r'[^a-zà-ÿ\s]', ' ', regex=True)   # keeps accented letters such as the e in cafe
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def assign_complaint_topics(texts, topic_bundle):
    # Topic id for each raw review text; -1 when none of its words are in the topic vocabulary.
    clean = normalize_text(texts)
    weights = topic_bundle['nmf'].transform(topic_bundle['tfidf'].transform(clean))
    topic_ids = weights.argmax(axis=1)
    topic_ids[weights.sum(axis=1) == 0] = -1
    return topic_ids


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def has_category(categories_text, category):
    if not isinstance(categories_text, str):
        return False
    return category in {c.strip() for c in categories_text.split(',')}


def find_competitors(latitude, longitude, radius_km, business_summary, category=None, open_only=False):
    # Businesses within radius_km of the point, nearest first. Businesses without usable coordinates are skipped.
    candidates = business_summary[business_summary['valid_coordinates'] == True].copy()
    if category:
        candidates = candidates[candidates['categories'].map(lambda text: has_category(text, category))]
    if open_only:
        candidates = candidates[candidates['is_open'] == 1]
    candidates['distance_km'] = haversine_km(latitude, longitude, candidates['latitude'], candidates['longitude'])
    nearby = candidates[candidates['distance_km'] <= radius_km]
    return nearby.sort_values('distance_km').reset_index(drop=True)


def analyze_location(latitude, longitude, radius_km, business_summary, business_topics, category=None, open_only=False):
    nearby = find_competitors(latitude, longitude, radius_km, business_summary, category, open_only)
    reviewed = nearby[nearby['reviews_analyzed'] > 0]
    total_reviews = int(reviewed['reviews_analyzed'].sum())

    # Compare with citywide businesses in the same category/status scope.
    # This avoids comparing cafes nearby with all restaurant types citywide.
    city_reference = business_summary.copy()
    if category:
        city_reference = city_reference[city_reference['categories'].map(lambda text: has_category(text, category))]
    if open_only:
        city_reference = city_reference[city_reference['is_open'] == 1]
    all_reviewed = city_reference[city_reference['reviews_analyzed'] > 0]
    all_reviews = max(int(all_reviewed['reviews_analyzed'].sum()), 1)
    city_rating = (all_reviewed['avg_rating'] * all_reviewed['reviews_analyzed']).sum() / all_reviews
    city_negative_pct = all_reviewed['negative_reviews'].sum() / all_reviews * 100

    average_rating, sentiment_pct = None, None
    if total_reviews > 0:
        average_rating = (reviewed['avg_rating'] * reviewed['reviews_analyzed']).sum() / total_reviews
        sentiment_pct = {kind.capitalize(): reviewed[kind + '_reviews'].sum() / total_reviews * 100
                         for kind in ['positive', 'neutral', 'negative']}

    # Complaint topics among nearby businesses compared with the whole city.
    local_topics = business_topics[business_topics['business_id'].isin(nearby['business_id'])]
    topics = (local_topics.groupby(['topic_id', 'topic_label'], as_index=False)['negative_reviews'].sum()
              .sort_values('negative_reviews', ascending=False).reset_index(drop=True))
    classified = int(topics['negative_reviews'].sum())
    city_topic_counts = (business_topics[business_topics['business_id'].isin(city_reference['business_id'])]
                         .groupby('topic_id')['negative_reviews'].sum())
    topics['share_pct'] = topics['negative_reviews'] / max(classified, 1) * 100
    city_topic_total = int(city_topic_counts.sum())
    if city_topic_total:
        topics['city_share_pct'] = topics['topic_id'].map(city_topic_counts / city_topic_total * 100).fillna(0)
    else:
        topics['city_share_pct'] = np.nan

    # For each topic, the nearby business with the most complaints in it and one example review.
    best = (local_topics.merge(nearby[['business_id', 'name', 'distance_km']], on='business_id')
            .sort_values('negative_reviews', ascending=False).drop_duplicates('topic_id'))
    examples = topics[['topic_id']].merge(best[['topic_id', 'topic_label', 'name', 'distance_km', 'example_stars', 'example_review']],
                                          on='topic_id')

    scope = category if category else 'restaurants and cafes'
    if open_only:
        scope = 'open ' + scope
    lines = [f'{len(nearby)} {scope} with usable coordinates within {radius_km:g} km.']
    opportunities = []
    if len(nearby) == 0:
        opportunities.append('No existing business in this radius has usable data, so these reviews cannot show whether '
                             'the area is under-served or simply has little demand.')
    elif total_reviews == 0:
        lines.append('None of them has reviews in the dataset, so there is no customer evidence for this area.')
    else:
        lines.append(f'{total_reviews:,} reviews from {len(reviewed)} of them. Weighted average rating {average_rating:.2f} '
                     f'(citywide comparison with the same category/status filter: {city_rating:.2f}).')
        lines.append(f"{sentiment_pct['Negative']:.1f}% of these reviews are 1-2 stars and {sentiment_pct['Positive']:.1f}% are 4-5 stars "
                     f'(matching citywide scope: {city_negative_pct:.1f}% are 1-2 stars).')
        if classified > 0:
            top_three = '; '.join(f'{r.topic_label} ({r.share_pct:.0f}%)' for r in topics.head(3).itertuples())
            lines.append(f'Most frequent complaint themes among {classified:,} classified complaints: {top_three}.')
        if total_reviews < MIN_REVIEWS_FOR_EVIDENCE:
            opportunities.append(f'Only {total_reviews} reviews are available nearby, so any pattern below is weak evidence.')
        if city_rating - average_rating >= 0.15:
            opportunities.append(f'Nearby competitors are rated {city_rating - average_rating:.2f} stars below the citywide average. '
                                 'Reading their reviews may show what customers feel is missing.')
        flagged = topics[(topics['share_pct'] - topics['city_share_pct'] >= 5) & (topics['negative_reviews'] >= 20)]
        for row in flagged.itertuples():
            opportunities.append(f'{row.topic_label} makes up {row.share_pct:.0f}% of nearby classified complaints versus '
                                 f'{row.city_share_pct:.0f}% citywide. Worth checking whether competitors here fall short on this theme.')
        if not opportunities:
            opportunities.append('Complaint themes and ratings around this location look close to the citywide pattern, '
                                 'so no local gap stands out in these reviews.')

    return {'competitors': nearby, 'n_competitors': len(nearby), 'n_with_reviews': len(reviewed),
            'total_reviews': total_reviews, 'average_rating': average_rating, 'sentiment_pct': sentiment_pct,
            'city_average_rating': city_rating, 'city_negative_pct': city_negative_pct,
            'topics': topics, 'examples': examples, 'summary_lines': lines, 'opportunities': opportunities}
