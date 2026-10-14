"""Design system for the BusinessLaunch AI app: CSS, page headers, cards, chart theme and game points."""
import html

import streamlit as st

PALETTE = {'teal': '#128C87', 'teal_dark': '#0F6F6B', 'navy': '#16324F', 'blue': '#387CB4', 'gold': '#D89B38',
           'red': '#C2514A', 'slate': '#94A3B8', 'muted': '#5B6B7F', 'mint': '#E3F4F1', 'bg': '#F6F9FA'}
VERDICT_STYLE = {'Promising': ('#E6F6F1', '#0F6F6B', '✅'), 'Proceed with caution': ('#FFF4E0', '#9A5B00', '⚠️'),
                 'High risk': ('#FDECEA', '#A33A32', '⛔')}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
[data-testid="stMetric"] {background: #FFFFFF;}
/* responsive metric cards: values shrink and labels wrap instead of being cut off on smaller screens */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {font-size: clamp(1.25rem, 2.05vw, 2.25rem) !important; overflow: visible !important; text-overflow: clip !important; white-space: nowrap;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p {white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.25;}
@media (max-width: 1100px) {.bl-hero h1 {font-size: 1.6rem;} .bl-hero {padding: 1.2rem 1.3rem;}}
[data-testid="stMetricLabel"] p {font-size: clamp(.66rem, .85vw, .82rem); color: #5B6B7F; font-weight: 600; text-transform: uppercase; letter-spacing: .02em; word-break: keep-all; overflow-wrap: normal; hyphens: none;}
[data-testid="stSidebarNav"] {padding-top: .5rem;}
.bl-hero {background: linear-gradient(120deg, #0F2A40 0%, #16324F 45%, #128C87 100%); color: #fff; border-radius: 1.1rem;
          padding: 1.6rem 1.9rem 1.4rem; margin-bottom: 1.1rem; position: relative; overflow: hidden;}
.bl-hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px; border-radius: 50%;
                background: rgba(255,255,255,.07);}
.bl-hero .eyebrow {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; opacity: .8; font-weight: 600;}
.bl-hero h1 {color: #fff; margin: .25rem 0 .35rem; font-size: 2.05rem; line-height: 1.15;}
.bl-hero p {color: rgba(255,255,255,.88); margin: 0; font-size: 1.02rem; max-width: 820px;}
.bl-chips {margin-top: .8rem; display: flex; gap: .4rem; flex-wrap: wrap;}
.bl-chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff; border-radius: 999px;
          padding: .14rem .65rem; font-size: .78rem; font-weight: 500;}
.bl-locbar {display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; align-items: center; background: #FFFFFF; border: 1px solid #D7E3E6;
            border-radius: .8rem; padding: .55rem .95rem; margin-bottom: 1rem; font-size: .9rem; color: #16324F;}
.bl-locbar b {color: #0F6F6B;}
.bl-pill {display: inline-block; border-radius: 999px; padding: .1rem .6rem; font-size: .78rem; font-weight: 600;}
.bl-section {margin: 1.4rem 0 .4rem; display: flex; align-items: baseline; gap: .6rem;}
.bl-section h3 {margin: 0 !important; padding: 0 !important; font-size: 1.45rem !important; font-weight: 700 !important;}
.bl-section .note {color: #5B6B7F; font-size: .92rem;}
.bl-card {background: #FFFFFF; border: 1px solid #D7E3E6; border-radius: .9rem; padding: 1rem 1.1rem; height: 100%;}
.bl-card h4 {margin: .2rem 0 .35rem; font-size: 1.02rem;}
.bl-card p {margin: 0; color: #5B6B7F; font-size: .9rem;}
.bl-card .icon {font-size: 1.5rem;}
.bl-feature {min-height: 200px;}
.bl-verdict {border-radius: 1rem; padding: 1.2rem 1.4rem; margin: .3rem 0 1rem; border: 1px solid rgba(0,0,0,.05);}
.bl-verdict .label {font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; opacity: .75;}
.bl-verdict .value {font-size: 2rem; font-weight: 800; line-height: 1.15; font-family: 'Plus Jakarta Sans', sans-serif;}
.bl-verdict p {margin: .35rem 0 0; font-size: 1rem;}
.bl-list {margin: 0; padding-left: 1.1rem;} .bl-list li {margin-bottom: .45rem; line-height: 1.4;}
.bl-footer {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #D7E3E6; color: #5B6B7F; font-size: .8rem;}
.bl-score {background: #173A55; border: 1px solid #2A4D69; border-radius: .8rem; padding: .6rem .8rem; margin-top: .4rem;}
.bl-score .pts {font-size: 1.5rem; font-weight: 800; color: #4FD1C5; line-height: 1.1;}
.bl-score .sub {font-size: .78rem; opacity: .8;}
.bl-game-option {font-size: 1.05rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, eyebrow='', chips=()):
    chips_html = ''.join(f'<span class="bl-chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(f'''<div class="bl-hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>
<p>{subtitle}</p>{f'<div class="bl-chips">{chips_html}</div>' if chips else ''}</div>''', unsafe_allow_html=True)


def section(title, note=''):
    st.markdown(f'<div class="bl-section"><h3>{html.escape(title)}</h3><span class="note">{html.escape(note)}</span></div>',
                unsafe_allow_html=True)


def feature_card(icon, title, text):
    st.markdown(f'<div class="bl-card bl-feature"><div class="icon">{icon}</div><h4>{html.escape(title)}</h4><p>{text}</p></div>',
                unsafe_allow_html=True)


def location_bar(name, lat, lon, radius, concept, cluster_name=None):
    label = f'★ {html.escape(name)}' if name else f'{lat:.4f}, {lon:.4f}'
    parts = [f'📍 <b>{label}</b>', f'{"🍽️" if concept == "Restaurant" else "☕"} {concept}', f'⭕ {radius:g} km radius']
    if cluster_name:
        parts.append(f'<span class="bl-pill" style="background:#E3F4F1;color:#0F6F6B">{html.escape(cluster_name)}</span>')
    st.markdown('<div class="bl-locbar">' + '<span>·</span>'.join(f'<span>{p}</span>' for p in parts) + '</div>',
                unsafe_allow_html=True)


def verdict_card(verdict, reason):
    bg, fg, icon = VERDICT_STYLE[verdict]
    st.markdown(f'''<div class="bl-verdict" style="background:{bg};color:{fg}"><div class="label">Recommendation</div>
<div class="value">{icon} {html.escape(verdict)}</div><p>{html.escape(reason)}</p></div>''', unsafe_allow_html=True)


def bullet_card(title, items, empty):
    lis = ''.join(f'<li>{html.escape(i)}</li>' for i in items) or f'<li>{html.escape(empty)}</li>'
    st.markdown(f'<div class="bl-card"><h4>{title}</h4><ul class="bl-list">{lis}</ul></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="bl-footer">BusinessLaunch AI · NTI Machine Learning & Data Analysis capstone · Data: Yelp Open '
                'Dataset (Philadelphia, snapshot January 2022). Popularity is measured by Yelp reviews, not revenue; '
                'money figures come from your assumptions.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- game points (shared across pages)
BADGES = [(10, '🥉 Rookie analyst'), (40, '🥈 Market scout'), (80, '🥇 Location strategist'), (150, '🏆 BusinessLaunch legend')]


def add_points(n, game=None):
    st.session_state['points'] = st.session_state.get('points', 0) + n
    if game:
        stats = st.session_state.setdefault('game_stats', {})
        stats[game] = stats.get(game, 0) + n


def current_badge(points):
    earned = [b for t, b in BADGES if points >= t]
    nxt = next(((t, b) for t, b in BADGES if points < t), None)
    return (earned[-1] if earned else None), nxt


def score_widget():
    points = st.session_state.get('points', 0)
    badge, nxt = current_badge(points)
    sub = f'{nxt[0] - points} pts to {nxt[1]}' if nxt else 'Top rank reached!'
    st.markdown(f'<div class="bl-score"><div class="sub">🎮 Your score</div><div class="pts">{points} pts</div>'
                f'<div class="sub">{badge or "Play a game to earn your first badge"} · {sub}</div></div>',
                unsafe_allow_html=True)
