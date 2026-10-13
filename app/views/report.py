"""Launch Report (all members): one explainable recommendation for the proposed location."""
import html
from datetime import date

import pandas as pd
import streamlit as st

from common import (current_location, current_saved_name, default_or_saved_assumptions, launch_report, load_market, load_reference,
                    load_saved, money, money_short, readable)
from businesslaunch import finance, recommend
from ui import bullet_card, footer, hero, location_bar, section, verdict_card

lat, lon, radius, concept = current_location()
hero('Launch Report', 'Everything the models found, turned into one decision: go, proceed with caution, or high risk.',
     eyebrow='All members · explainable recommendation',
     chips=['Opportunity & risk percentiles', 'Competitor complaints', 'Concept gaps', 'Monte Carlo finances', 'Alternatives'])
saved_name = current_saved_name()
index = load_market()
profile = index.profile(lat, lon, 1.0)
scores = recommend.location_scores(profile, load_reference())
assumptions, source = default_or_saved_assumptions(saved_name, concept, profile, scores)
report = launch_report(lat, lon, radius, concept, finance.as_dict(assumptions))
competitors, gaps, cluster_name = report['competitors_result'], report['gaps'], report['cluster']
sim_risk, f = report['simulation'], report['finance']
location_bar(saved_name, lat, lon, radius, concept, cluster_name)

if source == 'saved':
    st.success(f"Using the financial assumptions you saved for ★ {saved_name} on {load_saved()[saved_name].get('assumptions_saved_at', '?')}.",
               icon=':material/save:')
else:
    st.info('Using default finances scaled by local demand. Save the location (sidebar) and your own assumptions '
            '(Financial Simulator) to make this report yours.', icon=':material/info:')

v_col, s_col = st.columns([1.25, 1], gap='medium')
with v_col:
    verdict_card(report['verdict'], report['verdict_reason'])
with s_col:
    a1, a2 = st.columns(2)
    a1.metric('Opportunity', f"{report['scores']['opportunity']:.0f}/100", border=True)
    a2.metric('Market risk', f"{report['scores']['market_risk']:.0f}/100", border=True)
    a3, a4 = st.columns(2)
    a3.metric('Financial risk', f"{report['financial_risk']:.0f}/100", border=True,
              help='100 × share of simulations that do not pay back within 24 months')
    a4.metric('Overall risk', f"{report['overall_risk']:.0f}/100", border=True, help='60% market risk + 40% financial risk')
st.caption(f"{cluster_name} · {int(profile['n_businesses'])} restaurants/cafés within 1 km · {profile['open_share'] * 100:.0f}% still open · "
           f"review-weighted rating {profile['weighted_rating']:.2f} ★")

b1, b2, b3 = st.columns(3, gap='medium')
with b1:
    bullet_card('✅ Opportunities', report['strengths'], 'No clear advantage stands out.')
with b2:
    bullet_card('⚠️ Risks', report['risks'], 'No major risk flag.')
with b3:
    bullet_card('🎯 Recommended actions', report['actions'], 'Validate rent and staffing quotes, then re-run the simulator.')

tab_cust, tab_concept, tab_money, tab_alt, tab_compare = st.tabs([
    ':material/reviews: Customers', ':material/restaurant_menu: Concept fit', ':material/payments: Finances',
    ':material/explore: Better nearby areas', ':material/compare_arrows: Compare saved'])
with tab_cust:
    st.markdown('**What competitor reviews say**')
    for line in competitors['summary_lines']:
        st.write('• ' + readable(line))
    for line in competitors['opportunities']:
        st.info(readable(line), icon='💡')
with tab_concept:
    if len(gaps):
        view = gaps[['concept', 'local_count', 'expected_local', 'representation_ratio', 'city_median_reviews', 'gap_score']]
        cfg = {'concept': 'Concept', 'local_count': st.column_config.NumberColumn('Here', format='%d'),
               'expected_local': st.column_config.NumberColumn('Expected', format='%.1f'),
               'representation_ratio': st.column_config.NumberColumn('Here ÷ expected', format='%.2f'),
               'city_median_reviews': st.column_config.NumberColumn('City demand (median reviews)', format='%.0f'),
               'gap_score': st.column_config.ProgressColumn('Gap score', min_value=0, max_value=100, format='%.0f')}
        g1, g2 = st.columns(2, gap='medium')
        g1.markdown('**Under-represented here** · fewer than the city mix suggests')
        g1.dataframe(view.sort_values('gap_score', ascending=False).head(6), hide_index=True, width='stretch', column_config=cfg)
        g2.markdown('**Already concentrated here** · more than expected')
        g2.dataframe(view.sort_values('representation_ratio', ascending=False).head(6), hide_index=True, width='stretch', column_config=cfg)
with tab_money:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Profit / month', money_short(f['steady_monthly_net_profit']), border=True)
    m2.metric('Break-even', f"{f['break_even_customers_per_day']:.0f} / day", f'assumed {assumptions.customers_per_day:.0f}',
              delta_color='off', delta_arrow='off', border=True)
    m3.metric('Payback', f"month {f['payback_month']}" if f['payback_month'] else 'not within 36 m', border=True)
    m4.metric('Payback ≤ 24 m', f"{sim_risk['prob_payback_within_24m'] * 100:.0f}%", border=True)
    st.page_link('views/finance.py', label='Adjust the assumptions in the Financial Simulator', icon=':material/tune:')
with tab_alt:
    alternatives = report['alternatives']
    if len(alternatives):
        st.caption('Areas within 3 km with a clearly better opportunity-minus-risk balance.')
        for i, row in alternatives.iterrows():
            c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.1], vertical_alignment='center')
            c1.markdown(f"**{row['cluster_name']}**  \n{row['distance_km']:.1f} km away · {int(row['n_businesses'])} businesses within 1 km")
            c2.metric('Opportunity', f"{row['opportunity']:.0f}")
            c3.metric('Market risk', f"{row['market_risk']:.0f}")
            if c4.button('Analyse this area', key=f'alt_{i}', icon=':material/my_location:'):
                st.session_state.update(preset='Custom coordinates', lat=float(row['latitude']), lon=float(row['longitude']))
                st.rerun()
    else:
        st.success('No nearby area within 3 km scores clearly better than this one.', icon='👍')
with tab_compare:
    saved = load_saved()
    if len(saved) < 2:
        st.caption('Save two or more locations in the sidebar to compare them side by side.')
    else:
        rows = []
        for name, loc in saved.items():
            p = index.profile(loc['lat'], loc['lon'], 1.0)
            a, src = default_or_saved_assumptions(name, loc['concept'], p, recommend.location_scores(p, load_reference()))
            r = launch_report(loc['lat'], loc['lon'], loc['radius'], loc['concept'], finance.as_dict(a))
            rows.append({'location': ('▶ ' if name == saved_name else '') + name, 'concept': loc['concept'], 'verdict': r['verdict'],
                         'environment': r['cluster'], 'opportunity': r['scores']['opportunity'], 'market risk': r['scores']['market_risk'],
                         'overall risk': r['overall_risk'], 'competitors (1 km)': int(r['profile']['n_businesses']),
                         'monthly net profit': r['finance']['steady_monthly_net_profit'],
                         'P(payback ≤ 24 m)': r['simulation']['prob_payback_within_24m'] * 100,
                         'finances': 'saved' if src == 'saved' else 'default'})
        st.dataframe(pd.DataFrame(rows).sort_values('overall risk'), hide_index=True, width='stretch', column_config={
            'opportunity': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.0f'),
            'market risk': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.0f'),
            'overall risk': st.column_config.NumberColumn(format='%.0f'),
            'monthly net profit': st.column_config.NumberColumn(format='dollar'),
            'P(payback ≤ 24 m)': st.column_config.NumberColumn(format='%.0f%%')})
        st.caption('Sorted by overall risk (lowest first). ▶ = the location currently selected.')

# ---- downloadable report (Markdown and printable HTML)
title = saved_name or f'{lat:.4f}, {lon:.4f}'
md_lines = [f'# BusinessLaunch AI — launch report: {title}', f'_{date.today():%Y-%m-%d} · {concept} · radius {radius:g} km_', '',
            f"**Verdict: {report['verdict']}** — {report['verdict_reason']}", '',
            f"Market environment: {cluster_name}. Opportunity {scores['opportunity']:.0f}/100, market risk {scores['market_risk']:.0f}/100, "
            f"financial risk {report['financial_risk']:.0f}/100.", '', '## Opportunities', *[f'- {x}' for x in report['strengths']],
            '', '## Risks', *[f'- {x}' for x in report['risks']], '', '## Recommended actions', *[f'- {x}' for x in report['actions']],
            '', '## Customer reviews nearby', *[f'- {readable(x)}' for x in competitors['summary_lines']], '', '## Finances (your assumptions)',
            f"- Steady monthly revenue {money(f['steady_monthly_revenue'])}, net profit {money(f['steady_monthly_net_profit'])}",
            f"- Break-even {f['break_even_customers_per_day']:.0f} customers/day; payback month {f['payback_month']}",
            f"- Chance of payback within 24 months: {sim_risk['prob_payback_within_24m'] * 100:.0f}%", '',
            '_Yelp Open Dataset, Philadelphia, snapshot January 2022. Popularity is a review-count proxy; money figures are assumptions._']


def to_html(lines):
    body = []
    for line in lines:
        text = html.escape(line).replace('**', '')
        if line.startswith('# '):
            body.append(f'<h1>{text[2:]}</h1>')
        elif line.startswith('## '):
            body.append(f'<h2>{text[3:]}</h2>')
        elif line.startswith('- '):
            body.append(f'<li>{text[2:]}</li>')
        elif line:
            body.append(f'<p>{text.strip("_")}</p>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>Launch report</title><style>'
            'body{font-family:Inter,Arial,sans-serif;color:#16324F;max-width:760px;margin:40px auto;line-height:1.5}'
            'h1{color:#0F6F6B}h2{border-bottom:2px solid #E3F4F1;padding-bottom:4px;margin-top:28px}li{margin:4px 0}'
            '</style></head><body>' + '\n'.join(body) + '</body></html>')


section('Share this report')
d1, d2, _ = st.columns([1, 1, 2])
d1.download_button('Download as HTML (print to PDF)', to_html(md_lines).encode('utf-8'), 'launch_report.html', 'text/html',
                   icon=':material/description:', width='stretch')
d2.download_button('Download as Markdown', '\n'.join(md_lines).encode('utf-8'), 'launch_report.md', 'text/markdown',
                   icon=':material/download:', width='stretch')
footer()
