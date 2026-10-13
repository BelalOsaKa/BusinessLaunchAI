"""Neighbourhood market profiles (Member 3).

A *location profile* summarises every existing restaurant/café within a radius of a point:
supply (how many, how many still open), demand (check-ins, review volume), quality
(review-weighted rating, share of 1-2 star reviews), concept mix and complaint themes.
The same function builds the clustering dataset (profiles around grid points) and scores a
user's proposed location, so the app and the notebook always use identical features.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
CITY_HALL = (39.9526, -75.1636)

# Readable names for Member 2's NMF complaint topics, chosen from each topic's top words and
# representative 1-2 star reviews (see notebook 02, section M2-15). Topic ids are unchanged.
TOPIC_NAMES = {
    0: 'Service & staff attitude',
    1: 'Food taste & quality',
    2: 'Pizza quality',
    3: 'Orders, delivery & phone',
    4: 'Cheesesteaks & sandwiches',
    5: 'Wait times & table service',
}

# Tags that describe the business type rather than the concept; ignored in concept mix.
GENERIC_TAGS = {'Restaurants', 'Food', 'Cafes', 'Coffee & Tea', 'Nightlife', 'Bars'}

CLUSTER_FEATURES = [
    'log_businesses',        # supply: log(1 + restaurants/cafés within the radius)
    'open_share',            # survival: share still open at the snapshot
    'cafe_share',            # concept mix: share tagged Cafes / Coffee & Tea
    'nightlife_share',       # concept mix: share tagged Bars / Nightlife
    'avg_price',             # price level 1-4
    'weighted_rating',       # review-weighted star rating
    'negative_pct',          # share of 1-2 star reviews (pooled)
    'log_median_reviews',    # demand per business: log(1 + median review count)
    'checkin_activity',      # demand: mean log(1 + check-ins)
    'concept_diversity',     # Shannon entropy of concept tags
    'wait_complaint_share',  # share of complaints about waits / table service (topic 5)
    'delivery_complaint_share',  # share of complaints about orders / delivery (topic 3)
    'maturity_years',        # median years since first check-in
]

FEATURE_LABELS = {
    'log_businesses': 'Competition (log count)', 'open_share': 'Share still open',
    'cafe_share': 'Café share', 'nightlife_share': 'Bar / nightlife share', 'avg_price': 'Price level',
    'weighted_rating': 'Avg rating', 'negative_pct': '% 1-2 star reviews',
    'log_median_reviews': 'Reviews per business (log)', 'checkin_activity': 'Check-in activity (log)',
    'concept_diversity': 'Concept diversity', 'wait_complaint_share': 'Wait-time complaints',
    'delivery_complaint_share': 'Delivery complaints', 'maturity_years': 'Years on Yelp',
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def split_tags(text):
    return [t.strip() for t in str(text).split(',') if t.strip()] if isinstance(text, str) else []


def has_tag(text, tag):
    return tag in set(split_tags(text))


class MarketIndex:
    """Business universe plus a haversine BallTree and per-business complaint-topic counts."""

    def __init__(self, universe, business_topics):
        universe = universe.reset_index(drop=True).copy()
        self.universe = universe
        self.tags = universe['categories'].map(split_tags)
        self.is_cafe = self.tags.map(lambda t: bool({'Cafes', 'Coffee & Tea'} & set(t))).to_numpy()
        self.is_nightlife = self.tags.map(lambda t: bool({'Bars', 'Nightlife'} & set(t))).to_numpy()
        self.tree = BallTree(np.radians(universe[['latitude', 'longitude']].to_numpy(float)), metric='haversine')
        wide = (business_topics.pivot_table(index='business_id', columns='topic_id', values='negative_reviews',
                                            aggfunc='sum', fill_value=0)
                .reindex(universe['business_id']).fillna(0))
        self.topic_counts = wide.reindex(columns=sorted(TOPIC_NAMES), fill_value=0).to_numpy(float)

    def neighbours(self, lat, lon, radius_km):
        coords = np.radians(np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)]).astype(float))
        return self.tree.query_radius(coords, r=radius_km / EARTH_RADIUS_KM)

    def profile_from_indices(self, idx):
        u = self.universe
        if len(idx) == 0:
            return {f: np.nan for f in CLUSTER_FEATURES} | {'n_businesses': 0, 'n_open': 0, 'n_reviews': 0}
        reviews = u['reviews_analyzed'].to_numpy(float)[idx]
        total_reviews = reviews.sum()
        ratings = u['avg_rating'].to_numpy(float)[idx]
        valid = reviews > 0
        weighted_rating = (np.nansum(ratings[valid] * reviews[valid]) / total_reviews) if total_reviews else np.nan
        negative_pct = u['negative_reviews'].to_numpy(float)[idx].sum() / total_reviews * 100 if total_reviews else np.nan
        topics = self.topic_counts[idx].sum(axis=0)
        topic_total = topics.sum()
        concept_counts = Counter(t for tags in self.tags.iloc[idx] for t in tags if t not in GENERIC_TAGS)
        if concept_counts:
            p = np.array(list(concept_counts.values()), float)
            p /= p.sum()
            diversity = float(-(p * np.log(p)).sum())
        else:
            diversity = 0.0
        open_flags = u['is_open'].to_numpy(float)[idx]
        return {
            'n_businesses': int(len(idx)), 'n_open': int(np.nansum(open_flags)), 'n_reviews': int(total_reviews),
            'log_businesses': float(np.log1p(len(idx))),
            'open_share': float(np.nanmean(open_flags)),
            'cafe_share': float(self.is_cafe[idx].mean()),
            'nightlife_share': float(self.is_nightlife[idx].mean()),
            'avg_price': float(np.nanmean(u['price_range'].to_numpy(float)[idx])) if np.isfinite(u['price_range'].to_numpy(float)[idx]).any() else np.nan,
            'weighted_rating': float(weighted_rating),
            'negative_pct': float(negative_pct),
            'log_median_reviews': float(np.log1p(np.median(u['review_count'].to_numpy(float)[idx]))),
            'checkin_activity': float(np.mean(np.log1p(u['checkin_count'].to_numpy(float)[idx]))),
            'concept_diversity': diversity,
            'wait_complaint_share': float(topics[5] / topic_total * 100) if topic_total else np.nan,
            'delivery_complaint_share': float(topics[3] / topic_total * 100) if topic_total else np.nan,
            'maturity_years': float(np.nanmedian(u['years_on_yelp'].to_numpy(float)[idx]))
            if np.isfinite(u['years_on_yelp'].to_numpy(float)[idx]).any() else np.nan,
        }

    def profiles(self, points, radius_km=1.0):
        """Profiles for a DataFrame of points with latitude/longitude columns."""
        neighbourhoods = self.neighbours(points['latitude'], points['longitude'], radius_km)
        rows = [self.profile_from_indices(idx) for idx in neighbourhoods]
        result = pd.DataFrame(rows, index=points.index)
        result['distance_city_hall_km'] = haversine_km(points['latitude'], points['longitude'], *CITY_HALL)
        return result

    def profile(self, lat, lon, radius_km=1.0):
        return self.profiles(pd.DataFrame({'latitude': [lat], 'longitude': [lon]}), radius_km).iloc[0]

    def concept_gaps(self, lat, lon, radius_km=1.0, top_n=25, min_city_count=40):
        """Concept tags that are under- or over-represented around a point compared with the city.

        expected = local businesses x citywide share of the tag. A ratio below 1 means fewer local
        businesses carry the tag than the citywide mix would suggest (a possible gap), above 1 means
        the concept is already concentrated here. Demand is the citywide median review count of the tag.
        """
        idx = self.neighbours(lat, lon, radius_km)[0]
        city = Counter(t for tags in self.tags for t in tags if t not in GENERIC_TAGS)
        candidates = [t for t, c in city.most_common(top_n) if c >= min_city_count]
        local = Counter(t for tags in self.tags.iloc[idx] for t in tags if t not in GENERIC_TAGS)
        n_city, n_local = len(self.universe), len(idx)
        rows = []
        reviews = self.universe['review_count']
        ratings = self.universe['avg_rating']
        for tag in candidates:
            mask = self.tags.map(lambda t: tag in t).to_numpy()
            local_mask = np.zeros(n_city, bool)
            local_mask[idx] = True
            expected = n_local * city[tag] / n_city
            rows.append({'concept': tag, 'local_count': local.get(tag, 0), 'expected_local': expected,
                         'representation_ratio': (local.get(tag, 0) / expected) if expected else np.nan,
                         'city_count': city[tag],
                         'city_median_reviews': float(reviews[mask].median()),
                         'local_avg_rating': float(ratings[mask & local_mask].mean()) if (mask & local_mask).any() else np.nan,
                         'city_avg_rating': float(ratings[mask].mean())})
        gaps = pd.DataFrame(rows)
        if gaps.empty:
            return gaps
        demand_rank = gaps['city_median_reviews'].rank(pct=True)
        gaps['gap_score'] = ((1 - gaps['representation_ratio'].clip(upper=2) / 2) * 0.6 + demand_rank * 0.4) * 100
        return gaps.sort_values('gap_score', ascending=False).reset_index(drop=True)


def grid_points(universe, cell_km=0.5, min_businesses_in_cell=1):
    """One representative point per ~cell_km grid cell: the mean coordinate of the businesses in it."""
    lat0, lon0 = universe['latitude'].min(), universe['longitude'].min()
    lat_step = cell_km / 111.0
    lon_step = cell_km / (111.0 * np.cos(np.radians(39.95)))
    cell = (np.floor((universe['latitude'] - lat0) / lat_step).astype(int).astype(str) + '_' +
            np.floor((universe['longitude'] - lon0) / lon_step).astype(int).astype(str))
    points = (universe.assign(cell=cell).groupby('cell')
              .agg(latitude=('latitude', 'mean'), longitude=('longitude', 'mean'), cell_businesses=('business_id', 'size')))
    return points[points['cell_businesses'] >= min_businesses_in_cell].reset_index()
