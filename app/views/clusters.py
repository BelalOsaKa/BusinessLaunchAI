"""Market Clusters (Member 3): which market environment the location belongs to."""
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from common import (CLUSTER_COLORS, MEMBER3_DIR, cluster_for, current_location, current_saved_name, load_csv, load_market,
                    load_reference, location_scores, make_deck)
from businesslaunch.market import CLUSTER_FEATURES, FEATURE_LABELS
from ui import footer, hero, location_bar, section

# What each environment means for a new business owner (written from the cluster profiles in Part 3).
STORIES = {
    'Downtown': 'The busiest dining core: very high demand and visibility, but the densest competition and the highest '
                'turnover. Winning here needs a sharp concept and strong operations.',
    'Busy outer corridor': 'Commercial strips outside the centre with strong demand per business but weakly rated '
                           'competitors. Often the best opportunity-to-risk balance for a quality operator.',
    'Mixed': 'Established neighbourhood markets with average demand, ratings and survival. Success depends on fitting '
             'local tastes and building regulars.',
    'Quiet': 'Residential areas with few competitors and good survival but low activity. Cheaper rent, but you must '
             'create your own demand (delivery, destination concept).',
}


def story(name):
    return next((text for key, text in STORIES.items() if name.startswith(key)), '')


lat, lon, radius, concept = current_location()
hero('Market Environments', 'Philadelphia\'s 611 commercial areas fall into four market types. Which one is this?',
     eyebrow='Member 3 · unsupervised clustering',
     chips=['13 neighbourhood features', 'K-Means vs Ward vs GMM', 'Silhouette · Davies-Bouldin · bootstrap ARI', 'Survival back-test'])

index = load_market()
areas = load_reference().copy()
summary = load_csv(MEMBER3_DIR / 'cluster_summary.csv')
profile = index.profile(lat, lon, 1.0)   # clusters are built on 1 km profiles
cluster, name = cluster_for(profile)
scores = location_scores(profile)
location_bar(current_saved_name(), lat, lon, radius, concept, name)

colour = CLUSTER_COLORS[cluster % 8]
st.markdown(f'''<div class="bl-card" style="border-left:6px solid {colour};margin-bottom:1rem">
<div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:#5B6B7F;font-weight:700">Market environment</div>
<h3 style="margin:.2rem 0 .3rem">{name}</h3><p style="font-size:.98rem">{story(name)}</p></div>''', unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric('Opportunity', f"{scores['opportunity']:.0f}/100", border=True, help='50% demand + 25% dissatisfaction + 25% quality gap')
s2.metric('Market risk', f"{scores['market_risk']:.0f}/100", border=True, help='50% competition + 50% closure pressure')
s3.metric('Demand', f"{scores['demand']:.0f}", border=True, help='Check-ins and reviews per business (percentile)')
s4.metric('Competition', f"{scores['competition']:.0f}", border=True, help='Number of restaurants/cafés within 1 km (percentile)')
s5.metric('Closures', f"{scores['closure']:.0f}", border=True, help='Share of businesses no longer open (percentile)')
st.caption('All scores are percentiles against every Philadelphia commercial area (50 = typical).')

section('Map of market environments')
areas['color'] = areas['cluster'].map(lambda c: [int(CLUSTER_COLORS[c % 8][i:i + 2], 16) for i in (1, 3, 5)] + [175])
areas['opp'] = areas['opportunity'].round(0)
areas['risk'] = areas['market_risk'].round(0)
map_col, legend_col = st.columns([2, 1], gap='medium')
with map_col:
    st.pydeck_chart(make_deck(
        layers=[pdk.Layer('ScatterplotLayer', areas, get_position='[longitude, latitude]', get_fill_color='color', get_radius=190, pickable=True),
                pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                          get_fill_color=[214, 39, 40, 240], get_radius=150, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2)],
        initial_view_state=pdk.ViewState(latitude=39.98, longitude=-75.14, zoom=10.4),
        tooltip={'text': '{cluster_name}\nOpportunity {opp} · Risk {risk}\n{n_businesses} businesses within 1 km'}), height=450)
with legend_col:
    for row in summary.itertuples():
        c = CLUSTER_COLORS[int(row.cluster) % 8]
        st.markdown(f'''<div class="bl-card" style="margin-bottom:.55rem;padding:.65rem .8rem;border-left:5px solid {c}">
<b>{row.name}</b><p>{int(row.areas)} areas · median {row.businesses_median:.0f} businesses within 1 km · {row.open_share * 100:.0f}% still open
· {row.weighted_rating:.2f} ★</p></div>''', unsafe_allow_html=True)
    st.caption('Red = your location. Coordinates were not clustering inputs, yet the clusters follow the city\'s real geography.')

section('How this location compares', 'Standard deviations from the city average')
ref = areas[CLUSTER_FEATURES]
comparison = pd.DataFrame({
    'feature': [FEATURE_LABELS[f] for f in CLUSTER_FEATURES],
    'This location': [(profile[f] - ref[f].mean()) / ref[f].std() for f in CLUSTER_FEATURES],
    'Cluster average': [(areas.loc[areas['cluster'] == cluster, f].mean() - ref[f].mean()) / ref[f].std() for f in CLUSTER_FEATURES],
}).melt(id_vars='feature', var_name='series', value_name='z')
p1, p2 = st.columns([1.5, 1], gap='medium')
with p1:
    st.altair_chart(alt.Chart(comparison).mark_bar().encode(
        y=alt.Y('feature:N', title=None, sort=None, axis=alt.Axis(labelLimit=260)), x=alt.X('z:Q', title='← below average · above average →'),
        color=alt.Color('series:N', scale=alt.Scale(domain=['This location', 'Cluster average'], range=['#C2514A', '#CBD5E1']), title=None),
        yOffset=alt.YOffset('series:N', sort=['This location', 'Cluster average']),
        tooltip=['feature', 'series', alt.Tooltip('z', format='+.2f')]).properties(height=420, padding={'left': 36, 'right': 8}), width='stretch')
with p2:
    st.markdown('**Opportunity vs risk across the city**')
    scatter = alt.Chart(areas).mark_circle(size=28, opacity=.65).encode(
        x=alt.X('market_risk:Q', title='Market risk'), y=alt.Y('opportunity:Q', title='Opportunity'),
        color=alt.Color('cluster_name:N', title=None, scale=alt.Scale(range=CLUSTER_COLORS[:len(summary)]), legend=None),
        tooltip=['cluster_name', 'n_businesses', alt.Tooltip('opportunity', format='.0f'), alt.Tooltip('market_risk', format='.0f')])
    me = alt.Chart(pd.DataFrame({'x': [scores['market_risk']], 'y': [scores['opportunity']]})).mark_point(
        shape='diamond', size=320, color='#C2514A', filled=True, stroke='white', strokeWidth=1.5).encode(x='x', y='y')
    st.altair_chart((scatter + me).properties(height=380), width='stretch')
    st.caption('Red diamond = this location. Top-left = high opportunity, lower risk.')

backtest_file = MEMBER3_DIR / 'survival_backtest.csv'
if backtest_file.exists():
    section('Does the risk score predict survival?', 'Back-test on restaurants and cafés that opened in 2012–2019')
    bt = load_csv(backtest_file).copy()
    bt['Market risk'] = pd.qcut(bt['market_risk'], 3, labels=['Low', 'Medium', 'High'])
    bt['Demand'] = pd.qcut(bt['demand'], 3, labels=['Low', 'Medium', 'High'])
    b1, b2, b3 = st.columns([1, 1, .9], gap='medium')
    for col, key, colours in [(b1, 'Market risk', ['#128C87', '#D89B38', '#C2514A']), (b2, 'Demand', ['#CBD5E1', '#94A3B8', '#5B6B7F'])]:
        rates = bt.groupby(key, observed=True)['closed'].mean().mul(100).rename('closed').reset_index()
        col.altair_chart(alt.Chart(rates).mark_bar(size=46).encode(
            x=alt.X(f'{key}:N', sort=None, title=f'{key} tertile', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('closed:Q', title='% closed by 2022', scale=alt.Scale(domain=[0, 55])),
            color=alt.Color(f'{key}:N', scale=alt.Scale(range=colours), legend=None),
            tooltip=[alt.Tooltip('closed', format='.1f')]).properties(height=250), width='stretch')
    with b3:
        with st.container(border=True):
            st.markdown(f'**{len(bt):,} entrants tested**')
            st.write('Each was scored **without itself**, exactly like a new proposal. Higher market risk → more closures. '
                     'High demand does **not** mean safer: busy areas have more churn, so the verdict always weighs '
                     'opportunity against risk.')

with st.expander('How the clusters were chosen', icon=':material/science:'):
    selection = load_csv(MEMBER3_DIR / 'cluster_selection_metrics.csv')
    st.altair_chart(alt.Chart(selection).mark_line(point=True).encode(
        x=alt.X('k:O', title='Number of clusters', axis=alt.Axis(labelAngle=0)), y=alt.Y('silhouette:Q', title='Silhouette score'),
        color=alt.Color('method:N', title=None)).properties(height=260), width='stretch')
    st.dataframe(load_csv(MEMBER3_DIR / 'kmeans_stability.csv'), hide_index=True, width='stretch')
    st.write('Rule fixed in advance: k ≥ 4 (k = 2 only splits downtown from the rest), bootstrap ARI ≥ 0.8 and every cluster '
             '≥ 5% of areas, then the highest silhouette. K-Means scored best of the three methods and can assign new locations.')
    sens = MEMBER3_DIR / 'radius_sensitivity.csv'
    if sens.exists():
        st.dataframe(load_csv(sens), hide_index=True, width='stretch')
        st.caption('Re-clustering with a different neighbourhood radius: the partition largely survives at 1.5 km.')
footer()
