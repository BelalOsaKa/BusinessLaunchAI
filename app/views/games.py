"""Beat the AI: three short games built on the project's real data and trained models."""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from common import MEMBER1_DIR, MEMBER2_DIR, MEMBER3_DIR, load_csv, load_sentiment_model, make_deck
from businesslaunch.aspects import score_aspects
from businesslaunch.competitor_analysis import normalize_text
from businesslaunch.market import TOPIC_NAMES
from ui import BADGES, add_points, current_badge, footer, hero

hero('Beat the AI', 'Three quick games using real Philadelphia restaurants, real reviews and the models behind this app. '
     'Earn points, collect badges and see whether your instincts beat machine learning.',
     eyebrow='Play · learn how the models think',
     chips=['🔥 Higher or Lower', '🕵️ Complaint Detective', '📍 Pick the Spot'])
state = st.session_state
rng = np.random.default_rng()

# ---------------------------------------------------------------- scoreboard
points = state.get('points', 0)
badge, nxt = current_badge(points)
hl = state.setdefault('hl_stats', {'rounds': 0, 'you': 0, 'ai': 0, 'streak': 0, 'best_streak': 0})
det = state.setdefault('det_stats', {'rounds': 0, 'match': 0})
spot = state.setdefault('spot_stats', {'rounds': 0, 'best': 0})
s1, s2, s3, s4 = st.columns(4)
s1.metric('Score', f'{points} pts', badge or 'no badge yet', delta_color='off', delta_arrow='off', border=True,
          help=f'{nxt[0] - points} pts to {nxt[1]}' if nxt else 'Top rank reached')
s2.metric('You vs AI', f"{hl['you']} – {hl['ai']}", f"best streak {hl['best_streak']}", delta_color='off', delta_arrow='off',
          border=True, help='Higher or Lower: rounds won by you and by the regression model')
s3.metric('Detective', f"{det['match'] / det['rounds'] * 100 if det['rounds'] else 0:.0f}%", f"{det['rounds']} reviews",
          delta_color='off', delta_arrow='off', border=True, help='How often you matched the topic model')
s4.metric('Pick the Spot', f"{spot['best'] / spot['rounds'] * 100 if spot['rounds'] else 0:.0f}%", f"{spot['rounds']} maps",
          delta_color='off', delta_arrow='off', border=True, help='How often you found the best area')
if nxt:
    st.progress(min(points / nxt[0], 1.0), text=f'{nxt[0] - points} pts to {nxt[1]}')

game = st.segmented_control('Choose a game', ['🔥 Higher or Lower', '🕵️ Complaint Detective', '📍 Pick the Spot'],
                            default='🔥 Higher or Lower', key='game_choice', label_visibility='collapsed') or '🔥 Higher or Lower'


def card(title, lines, accent='#128C87', footer_html=''):
    body = ''.join(f'<p style="margin:.15rem 0">{line}</p>' for line in lines)
    st.markdown(f'<div class="bl-card" style="border-top:5px solid {accent}"><h4>{title}</h4>{body}{footer_html}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- 1. Higher or Lower
@st.cache_data
def popularity_pool():
    df = load_csv(MEMBER1_DIR / 'philadelphia_location_features.csv')
    df = df[df['split'].isin(['validation', 'test']) & (df['review_count'] >= 15)].copy()   # restaurants the model never trained on
    df['zip'] = pd.to_numeric(df['postal_code'], errors='coerce').astype('Int64').astype(str)
    df['tags'] = df['categories'].fillna('').map(lambda t: ', '.join([x.strip() for x in t.split(',')
                                                                      if x.strip() not in ('Restaurants', 'Food')][:3]))
    return df.reset_index(drop=True)


def new_pair(pool):
    while True:
        a, b = rng.choice(len(pool), 2, replace=False)
        ra, rb = pool.loc[a, 'review_count'], pool.loc[b, 'review_count']
        if max(ra, rb) / min(ra, rb) >= 1.4:
            return int(a), int(b)


if game == '🔥 Higher or Lower':
    pool = popularity_pool()
    st.markdown('#### Which restaurant has **more Yelp reviews**?')
    st.caption('Both are real restaurants from the held-out test areas, so the AI (the Member 1 regression model) never saw them. '
               '+10 for a correct answer, +5 bonus every 3 in a row.')
    if 'hl_pair' not in state:
        state.hl_pair, state.hl_answer = new_pair(pool), None
    a, b = state.hl_pair
    answered = state.hl_answer is not None
    cols = st.columns(2, gap='large')
    for col, idx, letter in zip(cols, (a, b), ('A', 'B')):
        r = pool.loc[idx]
        with col:
            reveal = ''
            if answered:
                winner = pool.loc[a, 'review_count'] > pool.loc[b, 'review_count']
                is_winner = (letter == 'A') == winner
                reveal = (f'<p style="margin-top:.6rem;font-size:1.5rem;font-weight:800;color:{"#0F6F6B" if is_winner else "#5B6B7F"}">'
                          f'{int(r["review_count"]):,} reviews {"🏆" if is_winner else ""}</p>'
                          f'<p style="color:#5B6B7F">AI estimate: {r["estimated_review_count"]:.0f}</p>')
            card(f'{letter} · {r["name"]}',
                 [f'🍽️ {r["tags"] or "Restaurant"}', f'⭐ {r["stars"]:.1f} · {"$" * int(r["price_range"]) if pd.notna(r["price_range"]) else "price n/a"}',
                  f'📍 ZIP {r["zip"]} · {r["years_since_first_checkin"]:.0f} years on Yelp' if pd.notna(r['years_since_first_checkin']) else f'📍 ZIP {r["zip"]}',
                  '🟢 still open' if r['is_open'] == 1 else '⚪ closed'],
                 accent='#128C87' if letter == 'A' else '#387CB4', footer_html=reveal)
            if not answered and st.button(f'{letter} has more reviews', key=f'hl_{letter}', type='primary', width='stretch'):
                state.hl_answer = letter
                truth = 'A' if pool.loc[a, 'review_count'] > pool.loc[b, 'review_count'] else 'B'
                ai = 'A' if pool.loc[a, 'estimated_review_count'] > pool.loc[b, 'estimated_review_count'] else 'B'
                hl['rounds'] += 1
                hl['ai'] += int(ai == truth)
                if letter == truth:
                    hl['you'] += 1
                    hl['streak'] += 1
                    hl['best_streak'] = max(hl['best_streak'], hl['streak'])
                    add_points(10 + (5 if hl['streak'] % 3 == 0 else 0), 'higher_lower')
                else:
                    hl['streak'] = 0
                state.hl_result = (letter == truth, ai == truth)
                st.rerun()
    if answered:
        you_ok, ai_ok = state.hl_result
        msg = {(True, True): 'You and the AI both got it! 🤝', (True, False): 'You beat the AI this round! 🎉',
               (False, True): 'The AI got this one. The model leans on business age and rating — try that too.',
               (False, False): 'Tricky one — neither of you got it.'}[(you_ok, ai_ok)]
        (st.success if you_ok else st.info)(msg + (f"  Streak: {hl['streak']} 🔥" if hl['streak'] >= 2 else ''))
        if st.button('Next pair →', type='primary', icon=':material/refresh:'):
            state.hl_pair, state.hl_answer = new_pair(pool), None
            st.rerun()

# ---------------------------------------------------------------- 2. Complaint Detective
@st.cache_data
def complaint_pool():
    df = load_csv(MEMBER2_DIR / 'business_complaint_topics.csv')
    df = df[df['example_review'].str.len().between(140, 650) & df['topic_id'].between(0, 5)]
    df['example_review'] = df['example_review'].str.strip().map(lambda t: t if t.endswith(('.', '!', '?', ')')) else t + '…')
    names = load_csv(MEMBER2_DIR / 'philadelphia_business_nlp_summary.csv')[['business_id', 'name']]
    return df.merge(names, on='business_id', how='left').reset_index(drop=True)


if game == '🕵️ Complaint Detective':
    pool = complaint_pool()
    st.markdown('#### What is this customer mainly complaining about?')
    st.caption('A real 1–2 ★ Philadelphia review. The AI is Member 2\'s NMF topic model, which grouped 127k complaints into six '
               'themes. Match its answer for +10 points.')
    if 'det_item' not in state:
        topic = int(rng.integers(0, 6))
        state.det_item, state.det_answer = int(rng.choice(pool.index[pool['topic_id'] == topic])), None
    row = pool.loc[state.det_item]
    st.markdown(f'''<div class="bl-card" style="border-left:5px solid #C2514A;font-size:1.02rem">
<p style="color:#16324F;font-size:1.02rem;font-style:italic">“{row['example_review']}”</p>
<p style="margin-top:.5rem">{'⭐' * int(row['example_stars'])} · {row['name'] if pd.notna(row['name']) else 'a Philadelphia restaurant'}</p></div>''',
                unsafe_allow_html=True)
    st.write('')
    answered = state.det_answer is not None
    opts = st.columns(3)
    for i, (topic_id, name) in enumerate(TOPIC_NAMES.items()):
        with opts[i % 3]:
            label = name
            if answered:
                label = ('✅ ' if topic_id == row['topic_id'] else ('❌ ' if topic_id == state.det_answer else '')) + name
            if st.button(label, key=f'det_{topic_id}', width='stretch', disabled=answered,
                         type='primary' if answered and topic_id == row['topic_id'] else 'secondary'):
                state.det_answer = topic_id
                det['rounds'] += 1
                if topic_id == row['topic_id']:
                    det['match'] += 1
                    add_points(10, 'detective')
                st.rerun()
    if answered:
        if state.det_answer == row['topic_id']:
            st.success(f"You think like the topic model! It filed this under **{TOPIC_NAMES[int(row['topic_id'])]}**. +10 pts")
        else:
            st.info(f"The topic model filed this under **{TOPIC_NAMES[int(row['topic_id'])]}** — reviews often mix several "
                    'complaints, and the model picks the strongest word pattern.')
        with st.container(border=True):
            st.markdown('**🔬 How the sentiment model reads it, sentence by sentence**')
            model = load_sentiment_model()
            p = float(model.predict_proba(normalize_text([row['example_review']]))[0, 1])
            st.progress(p, text=f'Overall: {p:.0%} positive')
            for r in score_aspects([row['example_review']], model).itertuples():
                st.markdown(f"{'🟢' if r.prob_positive >= .5 else '🔴'} **{r.aspect}** · {r.prob_positive:.0%} positive")
        if st.button('Next review →', type='primary', icon=':material/refresh:'):
            topic = int(rng.integers(0, 6))
            state.det_item, state.det_answer = int(rng.choice(pool.index[pool['topic_id'] == topic])), None
            st.rerun()

# ---------------------------------------------------------------- 3. Pick the Spot
@st.cache_data
def spot_pool():
    df = load_csv(MEMBER3_DIR / 'neighbourhood_profiles.csv').copy()
    df['balance'] = df['opportunity'] - df['market_risk']
    df['demand_level'] = pd.qcut(df['demand'], 3, labels=['Low', 'Medium', 'High']).astype(str)
    return df.reset_index(drop=True)


def fit_zoom(pins, height_ratio=2.8):
    """Zoom level that keeps every pin (plus its 1 km circle) inside a wide, short map."""
    span = max((pins['latitude'].max() - pins['latitude'].min()) * height_ratio,
               pins['longitude'].max() - pins['longitude'].min(), 0.02) + 0.06
    return float(np.clip(np.log2(360 / span) - 0.9, 8.5, 13))


def new_spots(pool):
    picks = [int(rng.choice(pool.index[pool['cluster'] == c])) for c in sorted(pool['cluster'].unique())]
    rng.shuffle(picks)
    return picks[:4]


if game == '📍 Pick the Spot':
    pool = spot_pool()
    st.markdown('#### You have one restaurant to open. Which area is the **best bet**?')
    st.caption('Four real Philadelphia areas (1 km around each pin). The AI ranks them by opportunity minus market risk — the '
               'score that predicted survival in our back-test. +15 for the best spot, +5 for the second best.')
    if 'spot_ids' not in state:
        state.spot_ids, state.spot_answer = new_spots(pool), None
    ids = state.spot_ids
    letters = ['A', 'B', 'C', 'D']
    answered = state.spot_answer is not None
    ranked = pool.loc[ids].sort_values('balance', ascending=False).index.tolist()
    pins = pool.loc[ids, ['latitude', 'longitude']].assign(letter=letters)
    st.pydeck_chart(make_deck(
        layers=[pdk.Layer('ScatterplotLayer', pins, get_position='[longitude, latitude]', get_fill_color=[18, 140, 135, 70],
                          get_radius=1000, stroked=True, get_line_color=[18, 140, 135, 200], line_width_min_pixels=1),
                pdk.Layer('TextLayer', pins, get_position='[longitude, latitude]', get_text='letter', get_size=26,
                          get_color=[22, 50, 79, 255], get_text_anchor='"middle"', get_alignment_baseline='"center"')],
        initial_view_state=pdk.ViewState(latitude=float(pins['latitude'].mean()), longitude=float(pins['longitude'].mean()),
                                         zoom=fit_zoom(pins))),
        height=320)
    cols = st.columns(4, gap='small')
    for col, idx, letter in zip(cols, ids, letters):
        r = pool.loc[idx]
        with col:
            extra = ''
            if answered:
                rank = ranked.index(idx) + 1
                extra = (f'<p style="margin-top:.5rem;font-weight:700;color:{"#0F6F6B" if rank == 1 else "#5B6B7F"}">'
                         f'{"🏆 " if rank == 1 else ""}AI rank #{rank} · balance {r["balance"]:+.0f}</p>'
                         f'<p style="color:#5B6B7F;font-size:.85rem">{r["cluster_name"]}</p>')
            card(f'Area {letter}', [f'🏪 {int(r["n_businesses"])} competitors', f'🟢 {r["open_share"] * 100:.0f}% still open',
                                    f'⭐ {r["weighted_rating"]:.2f} avg rating', f'😠 {r["negative_pct"]:.0f}% 1–2★ reviews',
                                    f'📈 demand: {r["demand_level"]}'], accent='#D89B38', footer_html=extra)
            if not answered and st.button(f'Open in {letter}', key=f'spot_{letter}', type='primary', width='stretch'):
                state.spot_answer = idx
                spot['rounds'] += 1
                rank = ranked.index(idx) + 1
                if rank == 1:
                    spot['best'] += 1
                    add_points(15, 'spot')
                elif rank == 2:
                    add_points(5, 'spot')
                st.rerun()
    if answered:
        rank = ranked.index(state.spot_answer) + 1
        best = pool.loc[ranked[0]]
        if rank == 1:
            st.success('Best spot! 🏆 +15 pts. ' + f"It is a **{best['cluster_name']}** area.")
        else:
            st.info(f"You picked the #{rank} spot{' (+5 pts)' if rank == 2 else ''}. The AI's best bet was Area "
                    f"{letters[ids.index(ranked[0])]}: a **{best['cluster_name']}** area with opportunity {best['opportunity']:.0f} "
                    f"and market risk {best['market_risk']:.0f}.")
        st.caption('Lesson from the data: high demand alone is not enough — busy areas also have the most competitors and closures.')
        if st.button('New map →', type='primary', icon=':material/refresh:'):
            state.spot_ids, state.spot_answer = new_spots(pool), None
            st.rerun()

st.divider()
with st.expander('Badges', icon=':material/military_tech:'):
    st.markdown('  \n'.join(f"{'✅' if points >= t else '⬜'} **{name}** — {t} pts" for t, name in BADGES))
    if st.button('Reset my scores', icon=':material/restart_alt:'):
        for key in ['points', 'game_stats', 'hl_stats', 'det_stats', 'spot_stats']:
            state.pop(key, None)
        st.rerun()
footer()
