"""Financial Simulator (Member 3): break-even, scenarios and Monte Carlo risk for the proposed business."""
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from common import (current_location, current_saved_name, finance_inputs, load_market, location_scores, money, money_short,
                    save_location,
                    saved_assumptions)
from businesslaunch import finance
from ui import footer, hero, location_bar, section

lat, lon, radius, concept = current_location()
hero('Financial Simulator', 'Do the numbers work? Break-even, three scenarios and 4,000 simulated futures for your assumptions.',
     eyebrow='Member 3 · scenario analysis & Monte Carlo',
     chips=['Break-even traffic', 'Ramp-up months', 'Pessimistic / base / optimistic', 'Monte Carlo risk', 'Sensitivity'])
saved_name = current_saved_name()
location_bar(saved_name, lat, lon, radius, concept)

profile = load_market().profile(lat, lon, 1.0)
scores = location_scores(profile)
multiplier = finance.traffic_multiplier(scores['demand'])
stored = saved_assumptions(saved_name)

with st.container(border=True):
    top_l, top_r = st.columns([3, 1], vertical_alignment='center')
    with top_l:
        st.markdown('**Your assumptions**')
        if stored is not None:
            st.caption(f'Loaded the assumptions you saved for ★ {saved_name}. Edit and save again to update the Launch Report.')
        else:
            st.caption(f"Defaults for a {concept.lower()} at the local price level ({profile['avg_price']:.1f} on Yelp's $–$$$$ scale). "
                       'Yelp has no revenue or cost data: replace these with real quotes before deciding.')
    use_demand = top_r.toggle('Scale traffic by local demand', value=True, disabled=stored is not None,
                              help=f"Demand score {scores['demand']:.0f}/100 → ×{multiplier:.2f} on the default customers per day")
    defaults = stored if stored is not None else finance.default_assumptions(concept, profile['avg_price'], multiplier if use_demand else 1.0)
    a = finance_inputs(defaults, key=f'fin_{concept}_{saved_name}_{defaults.customers_per_day:.0f}_{defaults.avg_ticket:.0f}')
    if saved_name:
        if st.button(f'Save these assumptions for “{saved_name}”', type='primary', icon=':material/save:'):
            save_location(saved_name, a)
            st.toast(f'Assumptions saved for {saved_name}. The Launch Report will use them.', icon='✅')
    else:
        with st.form('save_with_finance', border=False):
            n1, n2 = st.columns([3, 1], vertical_alignment='bottom')
            new_name = n1.text_input('Save this location and these assumptions as', placeholder='e.g. Fishtown corner unit')
            if n2.form_submit_button('Save', type='primary', icon=':material/save:', width='stretch') and new_name.strip():
                save_location(new_name.strip(), a)
                st.session_state['preset'] = '★ ' + new_name.strip()
                st.rerun()

summary = finance.summarize(a)
sim = finance.monte_carlo(a, n=4000)
risk = finance.risk_summary(sim, a)
safety = (a.customers_per_day / summary['break_even_customers_per_day'] - 1) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('Revenue / month', money_short(summary['steady_monthly_revenue']), 'at steady traffic', delta_color='off', delta_arrow='off', border=True)
k2.metric('Profit / month', money_short(summary['steady_monthly_net_profit']), f"{summary['steady_net_margin_pct']:.1f}% margin",
          delta_color='normal' if summary['steady_monthly_net_profit'] >= 0 else 'inverse', border=True)
k3.metric('Break-even', f"{summary['break_even_customers_per_day']:.0f} / day", f'{safety:+.0f}% margin',
          delta_color='normal' if safety >= 0 else 'inverse', border=True)
k4.metric('Payback', f"month {summary['payback_month']}" if summary['payback_month'] else f'> {a.horizon_months} months', border=True)
k5.metric('Payback ≤ 24 m', f"{risk['prob_payback_within_24m'] * 100:.0f}%", border=True,
          help='Share of 4,000 Monte Carlo simulations that recover the investment within 24 months')
if safety < 10:
    st.warning(f'Thin margin of safety: you need {summary["break_even_customers_per_day"]:.0f} customers a day and assume '
               f'{a.customers_per_day:.0f}. Lower fixed costs or raise the ticket before committing.', icon='⚠️')

tab_cash, tab_risk, tab_sens, tab_table = st.tabs([':material/show_chart: Cash over time', ':material/casino: Risk simulation',
                                                    ':material/tune: What matters most', ':material/table: Monthly table'])
with tab_cash:
    cases = {'Pessimistic': replace(a, customers_per_day=a.customers_per_day * 0.75, food_cost_pct=a.food_cost_pct + 3),
             'Base': a, 'Optimistic': replace(a, customers_per_day=a.customers_per_day * 1.2)}
    lines = pd.concat([finance.monthly_projection(case).assign(scenario=name) for name, case in cases.items()])
    chart = alt.Chart(lines).mark_line().encode(
        x=alt.X('month:Q', title='Month'), y=alt.Y('cumulative_cash:Q', title='Cumulative cash ($)', axis=alt.Axis(format='~s')),
        color=alt.Color('scenario:N', scale=alt.Scale(domain=['Pessimistic', 'Base', 'Optimistic'], range=['#C2514A', '#128C87', '#387CB4']),
                        title=None), tooltip=['scenario', 'month', alt.Tooltip('cumulative_cash', title='Cash', format='$,.0f')])
    zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[4, 4], color='#5B6B7F').encode(y='y')
    st.altair_chart((chart + zero).properties(height=330), width='stretch')
    st.caption('Pessimistic = −25% traffic and +3 pts food cost · optimistic = +20% traffic. Below the dashed line you have not yet recovered the investment.')
    table = finance.scenarios(a)[['customers_per_day', 'steady_monthly_revenue', 'steady_monthly_net_profit', 'year1_net_profit',
                                  'break_even_customers_per_day', 'payback_month', 'cash_at_horizon', 'roi_at_horizon_pct']]
    money_col = lambda label: st.column_config.NumberColumn(label, format='dollar')
    st.dataframe(table, width='stretch', column_config={
        'customers_per_day': st.column_config.NumberColumn('Customers / day', format='%.0f'),
        'steady_monthly_revenue': money_col('Monthly revenue'), 'steady_monthly_net_profit': money_col('Monthly profit'),
        'year1_net_profit': money_col('Year-1 profit'), 'break_even_customers_per_day': st.column_config.NumberColumn('Break-even / day', format='%.0f'),
        'payback_month': st.column_config.NumberColumn('Payback month', format='%d'), 'cash_at_horizon': money_col('Cash after 36 m'),
        'roi_at_horizon_pct': st.column_config.NumberColumn('ROI 36 m', format='%.0f%%')})
with tab_risk:
    r1, r2 = st.columns([1.6, 1], gap='medium')
    with r1:
        hist = sim.assign(cash=sim['cash_at_horizon'] / 1000, outcome=sim['cash_at_horizon'].map(lambda v: 'Loss' if v < 0 else 'Gain'))
        st.altair_chart(alt.Chart(hist).mark_bar().encode(
            x=alt.X('cash:Q', bin=alt.Bin(maxbins=50), title='Cash after 36 months ($ thousands)'), y=alt.Y('count()', title='Simulations'),
            color=alt.Color('outcome:N', scale=alt.Scale(domain=['Loss', 'Gain'], range=['#C2514A', '#128C87']), title=None)
        ).properties(height=320), width='stretch')
    with r2:
        st.metric('Pessimistic case (10th percentile)', money_short(risk['cash_at_horizon_p10']), border=True)
        st.metric('Typical case (median)', money_short(risk['cash_at_horizon_p50']), border=True)
        st.metric('Optimistic case (90th percentile)', money_short(risk['cash_at_horizon_p90']), border=True)
        st.caption(f"Loss-making steady state in {risk['prob_steady_state_loss'] * 100:.0f}% of simulations. Draws: traffic log-normal "
                   '(CV 25%), ticket ±10%, food cost ±3 pts.')
with tab_sens:
    base_cash = summary['cash_at_horizon']
    rows = []
    for field, label in [('customers_per_day', 'Customers per day'), ('avg_ticket', 'Average ticket'), ('food_cost_pct', 'Food cost %'),
                         ('labor_cost_month', 'Labour'), ('rent_month', 'Rent'), ('initial_investment', 'Investment')]:
        v = getattr(a, field)
        rows += [{'assumption': label, 'change': '−10%', 'delta': finance.summarize(replace(a, **{field: v * 0.9}))['cash_at_horizon'] - base_cash},
                 {'assumption': label, 'change': '+10%', 'delta': finance.summarize(replace(a, **{field: v * 1.1}))['cash_at_horizon'] - base_cash}]
    st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        y=alt.Y('assumption:N', title=None, sort=None), x=alt.X('delta:Q', title='Change in cash after 36 months ($)', axis=alt.Axis(format='~s')),
        color=alt.Color('change:N', scale=alt.Scale(domain=['−10%', '+10%'], range=['#D89B38', '#128C87']), title=None),
        tooltip=['assumption', 'change', alt.Tooltip('delta', format='$,.0f')]).properties(height=300), width='stretch')
    st.caption('Each bar changes one assumption by ±10% and keeps the rest fixed. Traffic and ticket size usually dominate.')
with tab_table:
    st.dataframe(finance.monthly_projection(a), hide_index=True, width='stretch', column_config={
        c: st.column_config.NumberColumn(format='dollar') for c in
        ['revenue', 'variable_costs', 'fixed_costs', 'operating_profit', 'tax', 'net_profit', 'cumulative_cash']} | {
        'traffic_pct': st.column_config.NumberColumn('Traffic %', format='%.0f%%'),
        'customers': st.column_config.NumberColumn('Customers', format='localized')})
footer()
