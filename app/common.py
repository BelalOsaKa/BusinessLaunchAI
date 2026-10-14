"""Shared loaders, location picker and helpers for every page of the BusinessLaunch AI app."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from businesslaunch import finance, recommend  # noqa: E402
from businesslaunch.location import PopularityModel  # noqa: E402
from businesslaunch.market import CLUSTER_FEATURES, MarketIndex  # noqa: E402
from businesslaunch.paths import MEMBER1_DIR, MEMBER2_DIR, MEMBER3_DIR  # noqa: E402

# Approximate centres of well-known Philadelphia dining areas (for quick demos).
PRESETS = {
    'Center City – City Hall': (39.9526, -75.1636),
    'Rittenhouse Square': (39.9496, -75.1718),
    'Old City': (39.9510, -75.1440),
    'Chinatown': (39.9555, -75.1550),
    'University City': (39.9530, -75.1930),
    'Fishtown': (39.9713, -75.1340),
    'Northern Liberties': (39.9637, -75.1415),
    'East Passyunk': (39.9290, -75.1652),
    'Manayunk': (40.0262, -75.2249),
    'Germantown': (40.0354, -75.1747),
    'Northeast – Frankford Ave': (40.0450, -75.0600),
    'Custom coordinates': None,
}
CLUSTER_COLORS = ['#128C87', '#387CB4', '#D89B38', '#C2514A', '#7A5BA6', '#4E9F3D', '#16324F', '#E07B39']

REQUIRED = {
    'Member 1': [MEMBER1_DIR / 'philadelphia_location_features.csv', MEMBER1_DIR / 'member1_model.joblib',
                 MEMBER1_DIR / 'test_metrics.csv'],
    'Member 2': [MEMBER2_DIR / 'philadelphia_business_nlp_summary.csv', MEMBER2_DIR / 'business_complaint_topics.csv',
                 MEMBER2_DIR / 'sentiment_model.pkl'],
    'Member 3': [MEMBER3_DIR / 'business_universe.csv', MEMBER3_DIR / 'neighbourhood_profiles.csv',
                 MEMBER3_DIR / 'market_cluster_model.joblib'],
}


def check_outputs():
    missing = [f'{who}: {p.relative_to(PROJECT_ROOT)}' for who, paths in REQUIRED.items() for p in paths if not p.exists()]
    if missing:
        st.error('Some model outputs are missing. Run the notebooks in `notebooks/` first:\n\n- ' + '\n- '.join(missing))
        st.stop()


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner='Loading models…')
def load_market():
    universe = pd.read_csv(MEMBER3_DIR / 'business_universe.csv')
    topics = pd.read_csv(MEMBER2_DIR / 'business_complaint_topics.csv')
    return MarketIndex(universe, topics)


@st.cache_resource(show_spinner=False)
def load_cluster_bundle():
    return joblib.load(MEMBER3_DIR / 'market_cluster_model.joblib')


@st.cache_resource(show_spinner=False)
def load_popularity_model():
    return PopularityModel(joblib.load(MEMBER1_DIR / 'member1_model.joblib'))


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return joblib.load(MEMBER2_DIR / 'sentiment_model.pkl')


DEFAULT_PRESET = 'Center City – City Hall'


def init_location_state():
    s = st.session_state
    s.setdefault('preset', DEFAULT_PRESET)
    s.setdefault('lat', PRESETS[DEFAULT_PRESET][0])
    s.setdefault('lon', PRESETS[DEFAULT_PRESET][1])
    s.setdefault('radius', 1.0)
    s.setdefault('concept', 'Restaurant')


def current_location():
    """(lat, lon, radius, concept) chosen in the sidebar (the router draws the sidebar before each page runs)."""
    init_location_state()
    s = st.session_state
    return float(s['lat']), float(s['lon']), float(s['radius']), s['concept']


SAVED_FILE = Path(__file__).with_name('saved_locations.json')
SAVED_PREFIX = '★ '


def _read_saved_file():
    try:
        return json.loads(SAVED_FILE.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def persist_to_file():
    """Write saved locations to disk only when the app runs on this computer.

    On a hosted deployment (Streamlit Community Cloud) one server is shared by every visitor and its disk is
    reset on each restart, so there saved locations live in the visitor's own session instead. Override with
    the environment variable BUSINESSLAUNCH_PERSIST=1 / 0.
    """
    forced = os.environ.get('BUSINESSLAUNCH_PERSIST')
    if forced in ('0', '1'):
        return forced == '1'
    try:
        host = st.context.headers.get('Host', '') or ''
    except Exception:  # no browser request (tests, bare mode)
        host = ''
    return host == '' or host.split(':')[0] in ('localhost', '127.0.0.1', '[::1]')


def load_saved():
    """Saved locations {name: {lat, lon, radius, concept, saved_at, assumptions?}}.

    Seeded from app/saved_locations.json (the three demo locations ship with the repo); kept per visitor in
    st.session_state, and also written back to the file when running locally.
    """
    state = st.session_state
    if persist_to_file():
        return _read_saved_file()
    if 'saved_locations' not in state:
        state['saved_locations'] = _read_saved_file()
    return dict(state['saved_locations'])


def write_saved(saved):
    if persist_to_file():
        SAVED_FILE.write_text(json.dumps(saved, indent=2), encoding='utf-8')
    else:
        st.session_state['saved_locations'] = dict(saved)


def current_saved_name():
    """Name of the saved location matching the current sidebar choice, or None."""
    s = st.session_state
    for name, loc in load_saved().items():
        if abs(loc['lat'] - s['lat']) < 1e-4 and abs(loc['lon'] - s['lon']) < 1e-4 and loc['concept'] == s['concept']:
            return name
    return None


def save_location(name, assumptions=None):
    s, saved = st.session_state, load_saved()
    entry = saved.get(name, {})
    entry.update({'lat': round(float(s['lat']), 5), 'lon': round(float(s['lon']), 5), 'radius': float(s['radius']),
                  'concept': s['concept'], 'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M')})
    if assumptions is not None:
        entry['assumptions'] = finance.as_dict(assumptions)
        entry['assumptions_saved_at'] = entry['saved_at']
    saved[name] = entry
    write_saved(saved)


def saved_assumptions(name):
    entry = load_saved().get(name or '', {})
    return finance.Assumptions(**entry['assumptions']) if 'assumptions' in entry else None


def location_picker():
    """Sidebar controls shared by all pages; the choice is kept in st.session_state."""
    init_location_state()
    state = st.session_state
    saved = load_saved()
    options = [SAVED_PREFIX + n for n in saved] + list(PRESETS)
    if state['preset'] not in options:
        state['preset'] = 'Custom coordinates'
    with st.sidebar:
        st.markdown('#### 📍 Proposed location')
        preset = st.selectbox('Area', options, index=options.index(state['preset']),
                              help='★ = your saved locations. Presets are well-known Philadelphia dining areas.')
        if preset != state['preset']:
            state['preset'] = preset
            if preset.startswith(SAVED_PREFIX):
                loc = saved[preset[len(SAVED_PREFIX):]]
                state['lat'], state['lon'], state['radius'], state['concept'] = loc['lat'], loc['lon'], loc['radius'], loc['concept']
            elif PRESETS[preset]:
                state['lat'], state['lon'] = PRESETS[preset]
        disabled = state['preset'] != 'Custom coordinates'
        state['lat'] = st.number_input('Latitude', 39.85, 40.15, float(state['lat']), 0.001, format='%.4f', disabled=disabled)
        state['lon'] = st.number_input('Longitude', -75.30, -74.95, float(state['lon']), 0.001, format='%.4f', disabled=disabled)
        state['radius'] = st.slider('Analysis radius (km)', 0.3, 3.0, float(state['radius']), 0.1)
        state['concept'] = st.radio('Concept', ['Restaurant', 'Café'], index=['Restaurant', 'Café'].index(state['concept']),
                                    horizontal=True)

        name = current_saved_name()
        st.divider()
        if name:
            entry = saved[name]
            st.success(f"★ Saved as **{name}**" + (' · finances saved' if 'assumptions' in entry else ''))
            if st.button('Remove from saved', key='unsave'):
                saved.pop(name)
                write_saved(saved)
                state['preset'] = 'Custom coordinates'
                st.rerun()
        else:
            with st.form('save_location', clear_on_submit=True, border=False):
                new_name = st.text_input('Save this location as', placeholder='e.g. Fishtown corner unit')
                if st.form_submit_button('💾 Save location') and new_name.strip():
                    save_location(new_name.strip())
                    state['preset'] = SAVED_PREFIX + new_name.strip()
                    st.rerun()
        st.caption('Saved locations appear at the top of the Area list and can be compared on the Launch Report page.'
                   + ('' if persist_to_file() else ' They are kept for this browser session.'))
    return state['lat'], state['lon'], state['radius'], state['concept']


def cluster_for(profile):
    bundle = load_cluster_bundle()
    features = profile[CLUSTER_FEATURES].to_frame().T.astype(float)
    cluster = int(bundle['model'].predict(features)[0])
    return cluster, bundle['names'][cluster]


@st.cache_data(show_spinner=False)
def load_reference():
    return pd.read_csv(MEMBER3_DIR / 'neighbourhood_profiles.csv')


def location_scores(profile):
    return recommend.location_scores(profile, load_reference())


def finance_inputs(defaults, key='fin'):
    """Editable financial assumptions; returns an Assumptions object."""
    d = finance.as_dict(defaults)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Demand & pricing**')
        d['customers_per_day'] = st.number_input('Customers per day (steady state)', 10.0, 2000.0, float(round(d['customers_per_day'])), 5.0, key=f'{key}_cust')
        d['avg_ticket'] = st.number_input('Average ticket ($)', 3.0, 300.0, float(d['avg_ticket']), 1.0, key=f'{key}_ticket')
        d['open_days_per_month'] = st.slider('Open days per month', 15, 31, int(d['open_days_per_month']), key=f'{key}_days')
        d['ramp_up_months'] = st.slider('Months to reach steady traffic', 1, 18, int(d['ramp_up_months']), key=f'{key}_ramp')
    with c2:
        st.markdown('**Operating costs**')
        d['food_cost_pct'] = st.slider('Food & drink cost (% of sales)', 15.0, 50.0, float(d['food_cost_pct']), 0.5, key=f'{key}_food')
        d['labor_cost_month'] = st.number_input('Labour ($/month)', 0.0, 300000.0, float(d['labor_cost_month']), 500.0, key=f'{key}_labor')
        d['rent_month'] = st.number_input('Rent ($/month)', 0.0, 100000.0, float(d['rent_month']), 250.0, key=f'{key}_rent')
        d['other_fixed_month'] = st.number_input('Utilities, insurance, other ($/month)', 0.0, 50000.0, float(d['other_fixed_month']), 250.0, key=f'{key}_other')
    with c3:
        st.markdown('**Investment & other**')
        d['initial_investment'] = st.number_input('Initial investment ($)', 0.0, 5_000_000.0, float(d['initial_investment']), 5000.0, key=f'{key}_inv')
        d['marketing_month'] = st.number_input('Marketing ($/month)', 0.0, 50000.0, float(d['marketing_month']), 100.0, key=f'{key}_mkt')
        d['card_and_platform_fees_pct'] = st.slider('Card & delivery-platform fees (%)', 0.0, 20.0, float(d['card_and_platform_fees_pct']), 0.5, key=f'{key}_fees')
        d['tax_rate_pct'] = st.slider('Tax on profit (%)', 0.0, 40.0, float(d['tax_rate_pct']), 1.0, key=f'{key}_tax')
    return finance.Assumptions(**d)


def money(x):
    return f'-${abs(x):,.0f}' if x < 0 else f'${x:,.0f}'


def make_deck(**kwargs):
    """pydeck Deck with an explicit Carto basemap (no API key needed; tiles load from the internet)."""
    import pydeck as pdk
    return pdk.Deck(map_provider='carto', map_style=pdk.map_styles.CARTO_LIGHT, **kwargs)


def default_or_saved_assumptions(name, concept, profile, scores):
    """(assumptions, source) — the user's saved finances for this location, else data-informed defaults."""
    stored = saved_assumptions(name)
    if stored is not None:
        return stored, 'saved'
    return finance.default_assumptions(concept, profile['avg_price'], finance.traffic_multiplier(scores['demand'])), 'default'


@st.cache_data(show_spinner='Building the launch report…', max_entries=32)
def launch_report(lat, lon, radius, concept, assumptions_dict):
    """Full recommendation for one location (cached; also used to compare saved locations)."""
    from businesslaunch.competitor_analysis import analyze_location
    index, reference = load_market(), load_reference()
    profile = index.profile(lat, lon, 1.0)
    _, cluster_name = cluster_for(profile)
    summary = load_csv(MEMBER2_DIR / 'philadelphia_business_nlp_summary.csv')
    topics = load_csv(MEMBER2_DIR / 'business_complaint_topics.csv')
    category = 'Restaurants' if concept == 'Restaurant' else 'Coffee & Tea'
    competitors = analyze_location(lat, lon, radius, summary, topics, category=category, open_only=True)
    gaps = index.concept_gaps(lat, lon, radius)
    assumptions = finance.Assumptions(**assumptions_dict)
    sim_risk = finance.risk_summary(finance.monte_carlo(assumptions, n=4000), assumptions)
    popularity = None
    if concept == 'Restaurant':
        popularity = load_popularity_model().estimate(lat=lat, lon=lon, stars=4.0, price_range=2, categories=['American (New)'],
                                                      attributes={'RestaurantsTakeOut': True, 'RestaurantsReservations': True})
    report = recommend.build_report(lat, lon, profile, cluster_name, reference, competitors, gaps, assumptions, sim_risk,
                                    popularity, concept)
    report.update({'profile': profile, 'competitors_result': competitors, 'gaps': gaps, 'assumptions': assumptions})
    return report


def money_short(x):
    """Compact money for metric cards: $950, $12.3k, $1.25M."""
    sign, v = ('-' if x < 0 else ''), abs(x)
    if v >= 1e6:
        return f'{sign}${v / 1e6:.2f}M'
    if v >= 1e4:
        return f'{sign}${v / 1e3:.1f}k'
    return f'{sign}${v:,.0f}'


@st.cache_data(show_spinner=False)
def _topic_label_map():
    from businesslaunch.market import TOPIC_NAMES
    topics = pd.read_csv(MEMBER2_DIR / 'business_complaint_topics.csv', usecols=['topic_id', 'topic_label']).drop_duplicates()
    return {row.topic_label: TOPIC_NAMES.get(int(row.topic_id), row.topic_label) for row in topics.itertuples()}


def readable(text):
    """Replace Member 2's raw NMF labels ('Topic 5: table, minutes, did') with readable theme names."""
    for raw, name in _topic_label_map().items():
        text = text.replace(raw, name)
    return text
