"""Unit tests for the shared modules.  Run:  python -m pytest tests   (or: python tests/test_core.py)"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from businesslaunch import finance, recommend  # noqa: E402
from businesslaunch.aspects import aspect_sentences, summarize_aspects  # noqa: E402
from businesslaunch.market import MarketIndex, haversine_km  # noqa: E402


def test_haversine_one_degree_longitude_at_equator():
    assert abs(haversine_km(0, 0, 0, 1) - 111.19) < 0.05


def test_break_even_matches_zero_operating_profit():
    a = finance.Assumptions(ramp_up_months=1)
    be = finance.break_even_customers_per_day(a)
    table = finance.monthly_projection(replace(a, customers_per_day=be))
    assert abs(table['operating_profit'].iloc[-1]) < 1e-6


def test_more_customers_never_reduce_cash():
    a = finance.Assumptions()
    cash = [finance.summarize(replace(a, customers_per_day=c))['cash_at_horizon'] for c in (60, 90, 120, 150)]
    assert all(x < y for x, y in zip(cash, cash[1:]))


def test_monte_carlo_matches_deterministic_model_without_noise():
    a = finance.Assumptions()
    sim = finance.monte_carlo(a, n=5, traffic_cv=1e-9, ticket_cv=0, cost_sd_pts=0, seed=1)
    expected = finance.summarize(a)
    assert np.allclose(sim['cash_at_horizon'], expected['cash_at_horizon'], rtol=1e-6)
    assert (sim['payback_month'] == expected['payback_month']).all()


def test_monte_carlo_is_reproducible():
    a = finance.Assumptions()
    pd.testing.assert_frame_equal(finance.monte_carlo(a, n=200, seed=7), finance.monte_carlo(a, n=200, seed=7))


def test_traffic_multiplier_is_bounded():
    assert finance.traffic_multiplier(0) == 0.6 and finance.traffic_multiplier(100) == 1.4
    assert finance.traffic_multiplier(float('nan')) == 1.0


def _toy_universe():
    universe = pd.DataFrame({
        'business_id': list('ABCD'), 'categories': ['Restaurants, Pizza', 'Cafes, Coffee & Tea', 'Restaurants, Bars', 'Restaurants'],
        'latitude': [39.9500, 39.9505, 39.9600, 40.0500], 'longitude': [-75.1600, -75.1605, -75.1600, -75.1600],
        'is_open': [1, 0, 1, 1], 'reviews_analyzed': [100, 50, 10, 20], 'avg_rating': [4.0, 3.0, 5.0, 2.0],
        'negative_reviews': [10, 20, 0, 15], 'price_range': [1, 2, np.nan, 3], 'review_count': [100, 50, 10, 20],
        'checkin_count': [10, 0, 5, 1], 'years_on_yelp': [5, 3, np.nan, 1]})
    topics = pd.DataFrame({'business_id': ['A', 'B', 'B'], 'topic_id': [5, 3, 5], 'negative_reviews': [4, 6, 2]})
    return MarketIndex(universe, topics)


def test_profile_aggregates_only_businesses_inside_radius():
    index = _toy_universe()
    p = index.profile(39.9500, -75.1600, radius_km=0.5)      # A and B only (C is ~1.1 km away)
    assert p['n_businesses'] == 2 and p['open_share'] == 0.5
    assert abs(p['weighted_rating'] - (4 * 100 + 3 * 50) / 150) < 1e-9
    assert abs(p['negative_pct'] - 30 / 150 * 100) < 1e-9
    assert abs(p['wait_complaint_share'] - 6 / 12 * 100) < 1e-9
    assert p['cafe_share'] == 0.5


def test_percentile_and_scores_are_in_range():
    ref = pd.DataFrame({'checkin_activity': [1, 2, 3], 'log_median_reviews': [1, 2, 3], 'negative_pct': [10, 20, 30],
                        'weighted_rating': [3, 4, 5], 'log_businesses': [1, 2, 3], 'open_share': [.5, .7, .9]})
    assert recommend.percentile(2, ref['checkin_activity']) == 50.0
    scores = recommend.location_scores(ref.iloc[2], ref)
    assert all(0 <= v <= 100 for v in scores.values())


def test_aspect_sentences_split_and_match():
    rows = aspect_sentences(['The food was delicious tonight. We waited 40 minutes for the bill!'])
    assert set(rows['aspect']) == {'Food & taste', 'Wait time', 'Price & value'}
    assert rows.groupby('aspect').size().max() == 1


def test_summarize_aspects_shares():
    scored = pd.DataFrame({'business_id': ['A', 'A', 'B'], 'aspect': ['Wait time'] * 3,
                           'prob_positive': [.9, .1, .2], 'positive': [1, 0, 0]})
    out = summarize_aspects(scored, group_col='business_id').set_index('business_id')
    assert out.loc['A', 'positive_share'] == 50 and out.loc['B', 'mentions'] == 1


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
    print(f'{len(tests)} tests passed')
