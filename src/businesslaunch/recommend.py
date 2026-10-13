"""Explainable launch recommendation (Member 3).

Combines the four analyses into one report for a proposed location:
  * market profile + cluster (Member 3 clustering),
  * competitor reviews and complaint themes (Member 2),
  * popularity what-if estimate (Member 1 regression, optional),
  * financial scenarios and Monte Carlo risk (Member 3 simulator).
Every score is a percentile against all Philadelphia grid locations, so "70" means the location
is higher than 70% of commercial areas in the city on that dimension. Weights are stated below
and kept simple on purpose; they are a transparent rule, not a fitted model.
"""
import numpy as np
import pandas as pd

from . import finance
from .market import FEATURE_LABELS, TOPIC_NAMES

OPPORTUNITY_WEIGHTS = {'demand': 0.5, 'dissatisfaction': 0.25, 'quality_gap': 0.25}
MARKET_RISK_WEIGHTS = {'competition': 0.5, 'closure': 0.5}
FINANCIAL_RISK_WEIGHT = 0.4   # share of overall risk that comes from the simulator


def percentile(value, reference):
    reference = np.asarray(reference, float)
    reference = reference[np.isfinite(reference)]
    if not np.isfinite(value) or len(reference) == 0:
        return np.nan
    return float(((reference < value).mean() + 0.5 * (reference == value).mean()) * 100)


def location_scores(profile, reference):
    """Percentile-based component scores for one profile against the grid reference profiles."""
    demand = np.nanmean([percentile(profile['checkin_activity'], reference['checkin_activity']),
                         percentile(profile['log_median_reviews'], reference['log_median_reviews'])])
    components = {
        'demand': demand,
        'dissatisfaction': percentile(profile['negative_pct'], reference['negative_pct']),
        'quality_gap': 100 - percentile(profile['weighted_rating'], reference['weighted_rating']),
        'competition': percentile(profile['log_businesses'], reference['log_businesses']),
        'closure': 100 - percentile(profile['open_share'], reference['open_share']),
    }
    components = {k: (50.0 if not np.isfinite(v) else v) for k, v in components.items()}
    components['opportunity'] = sum(components[k] * w for k, w in OPPORTUNITY_WEIGHTS.items())
    components['market_risk'] = sum(components[k] * w for k, w in MARKET_RISK_WEIGHTS.items())
    return components


def score_reference(reference):
    """Add opportunity / market-risk scores to every reference grid profile (used for alternatives)."""
    scores = pd.DataFrame([location_scores(row, reference) for _, row in reference.iterrows()], index=reference.index)
    return reference.join(scores[['opportunity', 'market_risk', 'demand', 'competition', 'closure']])


def _crowding(percentile_score):
    return 'among the most crowded' if percentile_score >= 99 else f'busier than {percentile_score:.0f}% of'


def verdict(opportunity, overall_risk, prob_loss):
    if overall_risk >= 70 or prob_loss > 0.5:
        return 'High risk', 'The combination of market pressure and the financial assumptions makes losses likely.'
    if opportunity >= 55 and overall_risk <= 50:
        return 'Promising', 'Demand signals are above average and the risk indicators are manageable.'
    return 'Proceed with caution', 'The location has a mix of positive and negative signals; validate the assumptions first.'


def build_report(lat, lon, profile, cluster, reference, competitor_result, gaps, assumptions,
                 sim_risk, popularity=None, concept='Restaurant'):
    """Assemble the recommendation. All inputs are computed by the caller (app or notebook)."""
    scores = location_scores(profile, reference)
    fin = finance.summarize(assumptions)
    financial_risk = (1 - sim_risk['prob_payback_within_24m']) * 100
    overall_risk = (1 - FINANCIAL_RISK_WEIGHT) * scores['market_risk'] + FINANCIAL_RISK_WEIGHT * financial_risk
    label, reason = verdict(scores['opportunity'], overall_risk, sim_risk['prob_steady_state_loss'])

    strengths, risks, actions = [], [], []
    if scores['demand'] >= 60:
        strengths.append(f"Customer activity is high: demand score {scores['demand']:.0f}/100 "
                         f"(check-ins and reviews per business above most of the city).")
    elif scores['demand'] <= 40:
        risks.append(f"Customer activity is low: demand score {scores['demand']:.0f}/100.")
    if scores['competition'] >= 70:
        level = 'Very dense' if scores['competition'] >= 85 else 'Dense'
        risks.append(f"{level} competition: {int(profile['n_businesses'])} restaurants/cafés within 1 km "
                     f"({_crowding(scores['competition'])} Philadelphia areas). A clear differentiator is required.")
    elif scores['competition'] <= 30:
        strengths.append(f"Few direct competitors ({int(profile['n_businesses'])} within 1 km).")
    if scores['closure'] >= 65:
        risks.append(f"Only {profile['open_share'] * 100:.0f}% of businesses here are still open "
                     f"(closure pressure {scores['closure']:.0f}/100).")
    if scores['quality_gap'] >= 60:
        strengths.append(f"Competitors are rated below most areas (avg {profile['weighted_rating']:.2f} stars): "
                         'room to win on quality.')
    if scores['dissatisfaction'] >= 60:
        strengths.append(f"{profile['negative_pct']:.1f}% of nearby reviews are 1-2 stars, above most areas: "
                         'customers are open to a better alternative.')

    topics = competitor_result.get('topics')
    if topics is not None and len(topics):
        over = topics[(topics['share_pct'] - topics['city_share_pct'] >= 3) & (topics['negative_reviews'] >= 20)]
        for row in over.head(2).itertuples():
            name = TOPIC_NAMES.get(int(row.topic_id), row.topic_label)
            actions.append(f"'{name}' makes up {row.share_pct:.0f}% of nearby complaints vs {row.city_share_pct:.0f}% "
                           'citywide: design operations to beat competitors on this point.')
    if gaps is not None and len(gaps):
        under = gaps[(gaps['representation_ratio'] < 0.7) & (gaps['city_median_reviews'] >= gaps['city_median_reviews'].median())]
        if len(under):
            names = ', '.join(under['concept'].head(3))
            actions.append(f'Under-represented concepts with solid citywide demand: {names}.')
        saturated = gaps[gaps['representation_ratio'] > 1.6].sort_values('representation_ratio', ascending=False)
        if len(saturated):
            risks.append('Already concentrated here: ' + ', '.join(saturated['concept'].head(3)) + '.')

    safety = (assumptions.customers_per_day / fin['break_even_customers_per_day'] - 1) * 100
    if safety < 10:
        risks.append(f"Thin margin of safety: break-even needs {fin['break_even_customers_per_day']:.0f} customers/day "
                     f"vs {assumptions.customers_per_day:.0f} assumed ({safety:+.0f}%).")
        actions.append('Reduce fixed costs (rent, staffing) or raise the average ticket before committing.')
    else:
        strengths.append(f"Assumed traffic is {safety:.0f}% above the break-even level "
                         f"({fin['break_even_customers_per_day']:.0f} customers/day).")
    if sim_risk['prob_payback_within_24m'] < 0.5:
        risks.append(f"Only {sim_risk['prob_payback_within_24m'] * 100:.0f}% of simulated outcomes recover the "
                     'investment within 24 months.')
    if popularity is not None and len(popularity):
        r3 = popularity.loc[popularity['years_on_yelp'] == 3, 'estimated_reviews']
        if len(r3):
            strengths.append(f"Member 1 model: a comparable restaurant here would collect about {r3.iloc[0]:.0f} "
                             'Yelp reviews in 3 years (a popularity proxy).')

    alternatives = suggest_alternatives(lat, lon, reference, scores)
    return {
        'verdict': label, 'verdict_reason': reason, 'concept': concept, 'cluster': cluster,
        'scores': scores, 'financial_risk': financial_risk, 'overall_risk': overall_risk,
        'finance': fin, 'simulation': sim_risk, 'strengths': strengths, 'risks': risks, 'actions': actions,
        'alternatives': alternatives,
        'profile_table': pd.DataFrame({'feature': [FEATURE_LABELS[k] for k in FEATURE_LABELS],
                                       'value': [profile[k] for k in FEATURE_LABELS]}),
    }


def suggest_alternatives(lat, lon, reference, scores, max_km=3.0, n=3):
    """Nearby grid areas with a better opportunity-minus-risk balance than the proposed point."""
    if 'opportunity' not in reference:
        return pd.DataFrame()
    from .market import haversine_km
    ref = reference.copy()
    ref['distance_km'] = haversine_km(lat, lon, ref['latitude'], ref['longitude'])
    ref['balance'] = ref['opportunity'] - ref['market_risk']
    current = scores['opportunity'] - scores['market_risk']
    better = ref[(ref['distance_km'] <= max_km) & (ref['distance_km'] > 0.4) & (ref['balance'] > current + 5)]
    cols = ['latitude', 'longitude', 'distance_km', 'cluster_name', 'opportunity', 'market_risk', 'n_businesses']
    cols = [c for c in cols if c in better]
    return better.sort_values('balance', ascending=False).head(n)[cols].reset_index(drop=True)
