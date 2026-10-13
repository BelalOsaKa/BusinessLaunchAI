"""Customer Reviews (Member 2): what customers of nearby competitors praise and complain about."""
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from common import MEMBER2_DIR, current_location, current_saved_name, load_csv, load_sentiment_model, make_deck, readable
from businesslaunch.aspects import score_aspects
from businesslaunch.competitor_analysis import COMPETITOR_COLUMNS, analyze_location, normalize_text
from businesslaunch.market import TOPIC_NAMES
from ui import footer, hero, location_bar, section

lat, lon, radius, concept = current_location()
hero('Customer Reviews', 'What do customers of nearby competitors love, and what makes them leave 1–2 star reviews?',
     eyebrow='Member 2 · NLP on 703,584 cleaned reviews',
     chips=['TF-IDF unigrams + bigrams', 'Tuned Logistic Regression · F1 0.944', 'NMF complaint topics', 'Aspect sentiment'])
location_bar(current_saved_name(), lat, lon, radius, concept)

summary = load_csv(MEMBER2_DIR / 'philadelphia_business_nlp_summary.csv')
topics = load_csv(MEMBER2_DIR / 'business_complaint_topics.csv')
options = ['All', 'Restaurants', 'Cafes', 'Coffee & Tea']
f1, f2 = st.columns([3, 1], vertical_alignment='bottom')
category = f1.segmented_control('Competitors to include', options,
                                default='Restaurants' if concept == 'Restaurant' else 'Coffee & Tea', key='rev_category') or 'All'
open_only = f2.toggle('Only businesses still open', value=True)
result = analyze_location(lat, lon, radius, summary, topics, category=None if category == 'All' else category, open_only=open_only)
competitors = result['competitors']
if competitors.empty:
    st.warning('No businesses with usable data in this radius. Increase the radius or change the category.', icon='🔍')
    footer()
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric('Competitors', result['n_competitors'], border=True)
m2.metric('Reviews analysed', f"{result['total_reviews']:,}", border=True)
if result['average_rating'] is not None:
    m3.metric('Review-weighted rating', f"{result['average_rating']:.2f} ★",
              f"{result['average_rating'] - result['city_average_rating']:+.2f} vs city", border=True)
    m4.metric('1–2 star reviews', f"{result['sentiment_pct']['Negative']:.1f}%",
              f"{result['sentiment_pct']['Negative'] - result['city_negative_pct']:+.1f} pts vs city", delta_color='inverse', border=True)

left, right = st.columns([1.7, 1], gap='medium')
with left:
    view = competitors.copy()
    view['color'] = view['avg_rating'].map(lambda r: [194, 81, 74, 210] if r < 3.3 else ([216, 155, 56, 210] if r < 4 else [18, 140, 135, 210]))
    view['rating'] = view['avg_rating'].round(2)
    view['top_complaint_topic'] = view['top_complaint_topic'].fillna('—').map(readable)
    st.pydeck_chart(make_deck(
        layers=[pdk.Layer('ScatterplotLayer', view, get_position='[longitude, latitude]', get_fill_color='color',
                          get_radius='22 + reviews_analyzed / 12', pickable=True, stroked=True, get_line_color=[255, 255, 255],
                          line_width_min_pixels=0.5),
                pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                          get_fill_color=[22, 50, 79, 240], get_radius=55, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2)],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14 - radius),
        tooltip={'text': '{name}\n{rating} ★ · {reviews_analyzed} reviews\nTop complaint: {top_complaint_topic}'}), height=400)
    st.caption('Teal ≥ 4.0 ★ · gold 3.3–4.0 · red < 3.3 · size = number of reviews · navy = your location')
with right:
    with st.container(border=True):
        st.markdown('**How nearby reviews split**')
        if result['sentiment_pct'] is not None:
            sent = pd.DataFrame({'sentiment': ['1–2 ★ negative', '3 ★ neutral', '4–5 ★ positive'],
                                 'share': [result['sentiment_pct'][k] for k in ['Negative', 'Neutral', 'Positive']]})
            st.altair_chart(alt.Chart(sent).mark_bar(height=26).encode(
                x=alt.X('share:Q', title='% of reviews', scale=alt.Scale(domain=[0, 100])), y=alt.Y('sentiment:N', sort=None, title=None, axis=alt.Axis(labelLimit=200)),
                color=alt.Color('sentiment:N', scale=alt.Scale(range=['#C2514A', '#94A3B8', '#128C87']), legend=None),
                tooltip=[alt.Tooltip('share', format='.1f')]).properties(height=150, padding={'left': 36, 'right': 8}), width='stretch')
        st.markdown('**Summary**')
        for line in result['summary_lines'][1:]:
            st.caption(readable(line))

if not result['topics'].empty:
    section('What customers complain about', 'NMF topics in 1–2 star reviews, nearby vs citywide')
    t = result['topics'].copy()
    t['theme'] = t['topic_id'].map(TOPIC_NAMES)
    t['gap'] = t['share_pct'] - t['city_share_pct']
    c_chart, c_text = st.columns([1.6, 1], gap='medium')
    with c_chart:
        long = t.melt(id_vars='theme', value_vars=['share_pct', 'city_share_pct'], var_name='where', value_name='share')
        long['where'] = long['where'].map({'share_pct': 'Nearby', 'city_share_pct': 'Citywide'})
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y('theme:N', title=None, sort=alt.SortField('share', order='descending'), axis=alt.Axis(labelLimit=260)),
            x=alt.X('share:Q', title='% of classified complaints'),
            color=alt.Color('where:N', scale=alt.Scale(domain=['Nearby', 'Citywide'], range=['#128C87', '#CBD5E1']), title=None),
            yOffset=alt.YOffset('where:N', sort=['Nearby', 'Citywide']),
            tooltip=['theme', 'where', alt.Tooltip('share', format='.1f')]).properties(height=300, padding={'left': 36, 'right': 8}), width='stretch')
    with c_text:
        worst = t.sort_values('gap', ascending=False).iloc[0]
        with st.container(border=True):
            st.markdown('**Biggest local pain point**')
            st.markdown(f"### {worst['theme']}")
            st.write(f"{worst['share_pct']:.0f}% of nearby complaints vs {worst['city_share_pct']:.0f}% citywide "
                     f"({worst['gap']:+.0f} pts). Beating competitors here is a concrete way to stand out.")
        for line in result['opportunities']:
            st.info(readable(line), icon='💡')
    with st.expander('Read representative complaints', icon=':material/format_quote:'):
        for row in result['examples'].head(5).itertuples():
            st.markdown(f"**{TOPIC_NAMES.get(int(row.topic_id), row.topic_label)}** · {row.name} · {row.distance_km:.2f} km · {row.example_stars}★")
            st.caption(row.example_review)

section('Likes and dislikes by aspect', 'Sentence-level sentiment for food, service, wait, price, ambience, cleanliness, delivery')
aspect_file = MEMBER2_DIR / 'business_aspect_sentiment.csv'
a_left, a_right = st.columns([1.2, 1], gap='medium')
with a_left:
    if aspect_file.exists():
        aspects = load_csv(aspect_file)

        def pooled(frame):
            frame = frame.assign(pos=frame['positive_share'] * frame['mentions'] / 100)
            g = frame.groupby('aspect')[['mentions', 'pos']].sum()
            return (g['pos'] / g['mentions'] * 100).rename('share')

        compare = pd.DataFrame({'Nearby': pooled(aspects[aspects['business_id'].isin(competitors['business_id'])]),
                                'Citywide': pooled(aspects)}).dropna().reset_index()
        long = compare.melt(id_vars='aspect', var_name='where', value_name='share')
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y('aspect:N', title=None, sort='x', axis=alt.Axis(labelLimit=260)), x=alt.X('share:Q', title='% positive mentions', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('where:N', scale=alt.Scale(domain=['Nearby', 'Citywide'], range=['#128C87', '#CBD5E1']), title=None),
            yOffset=alt.YOffset('where:N', sort=['Nearby', 'Citywide'])).properties(height=300, padding={'left': 36, 'right': 8}), width='stretch')
        weakest = (compare['Nearby'] - compare['Citywide']).idxmin()
        st.caption(f"Biggest local weakness: **{compare.loc[weakest, 'aspect']}** "
                   f"({compare.loc[weakest, 'Nearby']:.0f}% positive nearby vs {compare.loc[weakest, 'Citywide']:.0f}% citywide).")
    else:
        with st.container(border=True):
            st.markdown('**Area-level aspect scores**')
            st.write('These appear after Part 2 of the ML notebook is run with the full review file (section M2-16b). '
                     'Meanwhile, try the analyser on the right: it uses the same trained model.')
with a_right:
    with st.container(border=True):
        st.markdown('**🔬 Analyse any review**')
        text = st.text_area('Paste or write a review', 'The food was absolutely delicious and the portions were huge. '
                            'But we waited 45 minutes for a table and the waitress was rude.', height=110, label_visibility='collapsed')
        if text.strip():
            model = load_sentiment_model()
            probability = float(model.predict_proba(normalize_text([text]))[0, 1])
            st.progress(probability, text=f'Overall: {"positive" if probability >= .5 else "negative"} ({probability:.0%} positive)')
            scored = score_aspects([text], model)
            if scored.empty:
                st.caption('No aspect keywords found in this text.')
            for row in scored.itertuples():
                good = row.prob_positive >= 0.5
                st.markdown(f"{'🟢' if good else '🔴'} **{row.aspect}** · {row.prob_positive:.0%} positive  \n"
                            f"<span style='color:#5B6B7F;font-size:.85rem'>“{row.sentence}”</span>", unsafe_allow_html=True)

with st.expander(f'All nearby competitors ({len(competitors)})', icon=':material/table_view:'):
    st.dataframe(competitors[COMPETITOR_COLUMNS], hide_index=True, width='stretch', column_config={
        'avg_rating': st.column_config.NumberColumn('★', format='%.2f'), 'distance_km': st.column_config.NumberColumn('km', format='%.2f'),
        'negative_pct': st.column_config.ProgressColumn('% 1–2 ★', min_value=0, max_value=100, format='%.0f'),
        'positive_pct': st.column_config.ProgressColumn('% 4–5 ★', min_value=0, max_value=100, format='%.0f'),
        'neutral_pct': None, 'is_open': st.column_config.CheckboxColumn('Open')})
with st.expander('How the sentiment model was evaluated', icon=':material/insights:'):
    st.dataframe(load_csv(MEMBER2_DIR / 'nlp_model_comparison.csv'), hide_index=True, width='stretch')
    st.dataframe(load_csv(MEMBER2_DIR / 'nlp_test_metrics.csv'), hide_index=True, width='stretch')
    st.caption('Labels: 1–2 ★ negative, 4–5 ★ positive (3 ★ excluded from training). The split is by business, so test reviews '
               'come only from restaurants the model never saw.')
footer()
