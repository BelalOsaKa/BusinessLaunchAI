"""Restaurant / café financial simulator (Member 3).

Yelp has no revenue or cost data, so every money figure here comes from the user's assumptions.
The data only informs optional defaults: the local price level suggests a ticket size and the
local demand index (check-in activity versus the city median) can scale customer traffic.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

# Typical average spend per customer (USD) for Yelp price levels $ to $$$$; a starting point only.
TICKET_BY_PRICE = {1: 14.0, 2: 28.0, 3: 50.0, 4: 85.0}


@dataclass(frozen=True)
class Assumptions:
    customers_per_day: float = 120.0     # steady-state customers after ramp-up
    avg_ticket: float = 28.0             # average spend per customer (USD)
    open_days_per_month: float = 28.0
    food_cost_pct: float = 30.0          # cost of goods sold as % of revenue
    labor_cost_month: float = 22_000.0   # wages, payroll taxes
    rent_month: float = 7_000.0
    other_fixed_month: float = 4_000.0   # utilities, insurance, software, maintenance
    marketing_month: float = 1_500.0
    card_and_platform_fees_pct: float = 3.0
    initial_investment: float = 250_000.0  # fit-out, equipment, deposits, opening stock
    ramp_up_months: int = 6              # months to reach steady-state traffic
    ramp_start_pct: float = 55.0         # first-month traffic as % of steady state
    tax_rate_pct: float = 21.0           # applied to positive monthly profit only
    horizon_months: int = 36


def monthly_projection(a: Assumptions) -> pd.DataFrame:
    """Month-by-month revenue, costs, profit and cumulative cash including the initial investment."""
    months = np.arange(1, a.horizon_months + 1)
    if a.ramp_up_months > 1:
        ramp = np.clip(a.ramp_start_pct / 100 + (1 - a.ramp_start_pct / 100) * (months - 1) / (a.ramp_up_months - 1), 0, 1)
    else:
        ramp = np.ones_like(months, dtype=float)
    customers = a.customers_per_day * ramp * a.open_days_per_month
    revenue = customers * a.avg_ticket
    variable = revenue * (a.food_cost_pct + a.card_and_platform_fees_pct) / 100
    fixed = a.labor_cost_month + a.rent_month + a.other_fixed_month + a.marketing_month
    operating_profit = revenue - variable - fixed
    tax = np.where(operating_profit > 0, operating_profit * a.tax_rate_pct / 100, 0.0)
    net_profit = operating_profit - tax
    cumulative = -a.initial_investment + np.cumsum(net_profit)
    return pd.DataFrame({'month': months, 'traffic_pct': ramp * 100, 'customers': customers, 'revenue': revenue,
                         'variable_costs': variable, 'fixed_costs': fixed, 'operating_profit': operating_profit,
                         'tax': tax, 'net_profit': net_profit, 'cumulative_cash': cumulative})


def break_even_customers_per_day(a: Assumptions) -> float:
    """Daily customers needed for operating profit = 0 at steady state."""
    margin_per_customer = a.avg_ticket * (1 - (a.food_cost_pct + a.card_and_platform_fees_pct) / 100)
    fixed = a.labor_cost_month + a.rent_month + a.other_fixed_month + a.marketing_month
    if margin_per_customer <= 0:
        return float('inf')
    return fixed / margin_per_customer / a.open_days_per_month


def summarize(a: Assumptions) -> dict:
    table = monthly_projection(a)
    steady = table.iloc[-1]
    payback = table.loc[table['cumulative_cash'] >= 0, 'month']
    year1 = table[table['month'] <= 12]
    return {
        'steady_monthly_revenue': float(steady['revenue']),
        'steady_monthly_net_profit': float(steady['net_profit']),
        'steady_net_margin_pct': float(steady['net_profit'] / steady['revenue'] * 100) if steady['revenue'] else float('nan'),
        'year1_revenue': float(year1['revenue'].sum()),
        'year1_net_profit': float(year1['net_profit'].sum()),
        'break_even_customers_per_day': break_even_customers_per_day(a),
        'payback_month': int(payback.iloc[0]) if len(payback) else None,
        'cash_at_horizon': float(table['cumulative_cash'].iloc[-1]),
        'roi_at_horizon_pct': float(table['cumulative_cash'].iloc[-1] / a.initial_investment * 100),
        'lowest_cash_position': float(table['cumulative_cash'].min()),
    }


def scenarios(a: Assumptions, downside=0.75, upside=1.20) -> pd.DataFrame:
    """Pessimistic / base / optimistic traffic, with a small cost shock in the pessimistic case."""
    cases = {
        'Pessimistic': replace(a, customers_per_day=a.customers_per_day * downside, food_cost_pct=a.food_cost_pct + 3),
        'Base': a,
        'Optimistic': replace(a, customers_per_day=a.customers_per_day * upside),
    }
    return pd.DataFrame([{'scenario': name, 'customers_per_day': case.customers_per_day, **summarize(case)}
                         for name, case in cases.items()]).set_index('scenario')


def monte_carlo(a: Assumptions, n=4000, traffic_cv=0.25, ticket_cv=0.10, cost_sd_pts=3.0, seed=42) -> pd.DataFrame:
    """Simulate uncertain traffic, ticket size and food cost; returns one row per simulated outcome.

    Traffic is log-normal (cannot go negative), ticket size normal (floored at 30% of the base),
    food cost normal in percentage points. Draws are independent, a simplifying assumption.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + traffic_cv ** 2))
    traffic = a.customers_per_day * rng.lognormal(-sigma ** 2 / 2, sigma, n)
    ticket = np.maximum(a.avg_ticket * rng.normal(1, ticket_cv, n), a.avg_ticket * 0.3)
    food = np.clip(a.food_cost_pct + rng.normal(0, cost_sd_pts, n), 10, 80)
    # Same arithmetic as monthly_projection, vectorised over simulations (rows) and months (columns).
    ramp = monthly_projection(a)['traffic_pct'].to_numpy() / 100
    revenue = (traffic[:, None] * ramp[None, :] * a.open_days_per_month) * ticket[:, None]
    variable = revenue * ((food[:, None] + a.card_and_platform_fees_pct) / 100)
    fixed = a.labor_cost_month + a.rent_month + a.other_fixed_month + a.marketing_month
    operating = revenue - variable - fixed
    net = operating - np.where(operating > 0, operating * a.tax_rate_pct / 100, 0.0)
    cumulative = -a.initial_investment + np.cumsum(net, axis=1)
    reached = cumulative >= 0
    payback = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, np.nan)
    return pd.DataFrame({'customers_per_day': traffic, 'avg_ticket': ticket, 'food_cost_pct': food,
                         'cash_at_horizon': cumulative[:, -1], 'payback_month': payback,
                         'steady_monthly_net_profit': net[:, -1]})


def risk_summary(sim: pd.DataFrame, a: Assumptions) -> dict:
    paid_back = sim['payback_month'].notna()
    return {
        'prob_payback_within_horizon': float(paid_back.mean()),
        'prob_payback_within_24m': float((sim['payback_month'].fillna(np.inf) <= 24).mean()),
        'prob_steady_state_loss': float((sim['steady_monthly_net_profit'] < 0).mean()),
        'cash_at_horizon_p10': float(sim['cash_at_horizon'].quantile(0.10)),
        'cash_at_horizon_p50': float(sim['cash_at_horizon'].quantile(0.50)),
        'cash_at_horizon_p90': float(sim['cash_at_horizon'].quantile(0.90)),
        'median_payback_month': float(sim.loc[paid_back, 'payback_month'].median()) if paid_back.any() else None,
        'horizon_months': a.horizon_months,
    }


def traffic_multiplier(demand_score):
    """Map a 0-100 demand percentile to a bounded traffic multiplier between 0.6x and 1.4x.

    The demand score is based on activity *per business* (check-ins, reviews), so it already
    accounts for traffic being shared among competitors. The +/-40% range is an assumption.
    """
    score = 50.0 if demand_score is None or not np.isfinite(demand_score) else float(demand_score)
    return 0.6 + 0.8 * np.clip(score, 0, 100) / 100


def default_assumptions(concept='Restaurant', price_level=2.0, demand_multiplier=1.0):
    """Concept-specific starting assumptions; traffic is multiplied by `demand_multiplier`."""
    level = int(np.clip(round(price_level if np.isfinite(price_level) else 2), 1, 4))
    if concept == 'Café':
        base = Assumptions(customers_per_day=220, avg_ticket=min(TICKET_BY_PRICE[level], 12.0), food_cost_pct=28,
                           labor_cost_month=14_000, rent_month=4_500, other_fixed_month=2_500, marketing_month=800,
                           initial_investment=120_000)
    else:
        base = Assumptions(avg_ticket=TICKET_BY_PRICE[level])
    return replace(base, customers_per_day=base.customers_per_day * float(demand_multiplier))


def as_dict(a: Assumptions) -> dict:
    return asdict(a)
