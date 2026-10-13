"""Home: what the tool does, the selected location at a glance, and where to go next."""
import json

import pandas as pd
import pydeck as pdk
import streamlit as st

from common import (CLUSTER_COLORS, MEMBER1_DIR, MEMBER2_DIR, MEMBER3_DIR, cluster_for, current_location, current_saved_name,
                    load_csv, load_market, location_scores, make_deck)
from ui import feature_card, footer, hero, location_bar, section

lat, lon, radius, concept = current_location()
hero('Should I open a restaurant or café here?',
     'BusinessLaunch AI reads 14,570 Philadelphia businesses, ~700k Yelp reviews and 1.29M check-ins to show the '
     '<b>competition</b>, <b>customer expectations</b>, <b>market type</b> and <b>financial risk</b> of any location — '
     'before you sign a lease.',
     eyebrow='Philadelphia · Yelp Open Dataset · NTI ML capstone',
     chips=['Regression', 'NLP sentiment', 'Topic modelling', 'Clustering', 'Monte Carlo', 'Explainable scoring'])

index = load_market()
profile = index.profile(lat, lon, radius)
profile_1km = index.profile(lat, lon, 1.0)
cluster, cluster_name = cluster_for(profile_1km)
scores = location_scores(profile_1km)
location_bar(current_saved_name(), lat, lon, radius, concept, cluster_name)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Competitors', f"{int(profile['n_businesses']):,}", f'within {radius:g} km', delta_color='off', delta_arrow='off',
          border=True, help='Restaurants and cafés (open or closed) in the analysis radius')
c2.metric('Still open', f"{profile['open_share'] * 100:.0f}%", border=True, help='Share of those businesses open at the snapshot')
c3.metric('Avg rating', f"{profile['weighted_rating']:.2f} ★", border=True, help='Review-weighted Yelp rating')
c4.metric('Opportunity', f"{scores['opportunity']:.0f}/100", border=True, help='Percentile vs all Philadelphia areas (demand, dissatisfaction, quality gap)')
c5.metric('Market risk', f"{scores['market_risk']:.0f}/100", border=True, help='Percentile vs all areas (competition, closures)')

left, right = st.columns([2.1, 1], gap='large')
with left:
    areas = load_csv(MEMBER3_DIR / 'neighbourhood_profiles.csv').copy()
    areas['color'] = areas['cluster'].map(lambda c: [int(CLUSTER_COLORS[c % 8][i:i + 2], 16) for i in (1, 3, 5)] + [150])
    areas['opp'] = areas['opportunity'].round(0)
    layers = [
        pdk.Layer('ScatterplotLayer', areas, get_position='[longitude, latitude]', get_fill_color='color', get_radius=170, pickable=True),
        pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                  get_fill_color=[214, 39, 40, 30], get_radius=radius * 1000, stroked=True, get_line_color=[214, 39, 40, 160],
                  line_width_min_pixels=1),
        pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                  get_fill_color=[214, 39, 40, 240], get_radius=110, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2),
    ]
    st.pydeck_chart(make_deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=11.3),
                              tooltip={'text': '{cluster_name}\nOpportunity {opp}/100\n{n_businesses} businesses within 1 km'}),
                    height=430)
    summary = load_csv(MEMBER3_DIR / 'cluster_summary.csv')
    st.markdown(' '.join(f'<span class="bl-pill" style="background:{CLUSTER_COLORS[int(r.cluster) % 8]};color:white">{r.name}</span>'
                         for r in summary.itertuples()) + ' <span class="bl-pill" style="background:#D62728;color:white">● your location</span>',
                unsafe_allow_html=True)
with right:
    with st.container(border=True):
        st.markdown(f'**This is a _{cluster_name}_ area.**')
        verdict_hint = ('High demand but tough competition — you will need a clear edge.' if scores['market_risk'] >= 65
                        else 'Lower competition — check that demand is strong enough.' if scores['opportunity'] < 45
                        else 'A balanced market — worth a closer look.')
        st.write(verdict_hint)
        st.page_link('views/report.py', label='See the full launch report', icon=':material/task_alt:')
        st.page_link('views/finance.py', label='Run the numbers', icon=':material/payments:')
        st.page_link('views/reviews.py', label='Read what customers complain about', icon=':material/reviews:')
    with st.container(border=True):
        st.markdown('**🎮 Can you beat the AI?**')
        st.write('Guess which restaurant is more popular, detect complaint themes and pick the best spot — '
                 'against our trained models.')
        st.page_link('views/games.py', label='Play now', icon=':material/sports_esports:')

section('How it works', 'Five steps from a pin on the map to a decision')
cards = [('📍', 'Competition', 'Competitors within 0.5–2 km and a popularity estimate for your concept (Member 1 · regression).', 'views/location.py'),
         ('💬', 'Customers', 'What nearby customers praise and complain about (Member 2 · NLP + topics).', 'views/reviews.py'),
         ('🗺️', 'Market type', 'Which of 4 Philadelphia market environments this is (Member 3 · clustering).', 'views/clusters.py'),
         ('💰', 'Finances', 'Break-even, scenarios and Monte Carlo risk for your assumptions (Member 3).', 'views/finance.py'),
         ('📋', 'Decision', 'An explainable verdict, actions and better nearby areas (all members).', 'views/report.py')]
for col, (icon, title, text, target) in zip(st.columns(5), cards):
    with col:
        feature_card(icon, title, text)
        st.page_link(target, label=f'Open {title.lower()}', icon=':material/arrow_forward:')

section('Model quality', 'Measured on held-out data the models never saw')
m1 = json.loads((MEMBER1_DIR / 'run_summary.json').read_text())
m2 = load_csv(MEMBER2_DIR / 'nlp_test_metrics.csv').iloc[0]
m3 = json.loads((MEMBER3_DIR / 'run_summary.json').read_text())
q1, q2, q3, q4 = st.columns(4)
q1.metric('Popularity R²', f"{m1['test_metrics']['R2']:.2f}", f"MAE {m1['test_metrics']['MAE']:.0f} reviews",
          delta_color='off', delta_arrow='off', border=True, help=f"{m1['selected_model']} with a spatial train/validation/test split")
q2.metric('Sentiment F1', f"{m2['f1_macro']:.3f}", f"accuracy {m2['accuracy']:.3f}", delta_color='off', delta_arrow='off', border=True,
          help=str(m2['model']))
q3.metric('Market types', f"{m3['k']}", f"silhouette {m3['silhouette']:.2f}", delta_color='off', delta_arrow='off', border=True,
          help=f"K-Means, bootstrap stability ARI {m3['bootstrap_ARI']:.2f}")
backtest = MEMBER3_DIR / 'survival_backtest.csv'
if backtest.exists():
    bt = load_csv(backtest)
    tert = pd.qcut(bt['market_risk'], 3, labels=False)
    q4.metric('Closures by risk', f"{bt.loc[tert == 0, 'closed'].mean() * 100:.0f}→{bt.loc[tert == 2, 'closed'].mean() * 100:.0f}%",
              'low → high risk areas', delta_color='off', delta_arrow='off', border=True,
              help=f'{len(bt):,} restaurants/cafés that opened 2012–2019')
footer()
