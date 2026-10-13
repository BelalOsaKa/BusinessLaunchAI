"""Location Intelligence (Member 1): competition around the location and the popularity regression model."""
import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from common import (MEMBER1_DIR, current_location, current_saved_name, load_csv, load_popularity_model, make_deck)
from businesslaunch.market import haversine_km
from ui import footer, hero, location_bar, section

lat, lon, radius, concept = current_location()
hero('Location Intelligence', 'How crowded is this spot, and how popular could a new restaurant here become?',
     eyebrow='Member 1 · competition & popularity regression',
     chips=['Haversine BallTree features', 'Spatial train/validation/test split', 'HistGradientBoosting', 'Calibrated 80% range'])
location_bar(current_saved_name(), lat, lon, radius, concept)

model = load_popularity_model()
restaurants = load_csv(MEMBER1_DIR / 'philadelphia_location_features.csv').copy()
restaurants['postal_code'] = pd.to_numeric(restaurants['postal_code'], errors='coerce').astype('Int64').astype(str)
loc = model.location_features(lat, lon)
city = restaurants[['competitors_0p5km', 'competitors_1p0km', 'competitors_2p0km', 'open_competitors_1km']].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Restaurants ≤ 0.5 km', loc['competitors_0p5km'], f"{loc['competitors_0p5km'] - city['competitors_0p5km']:+.0f} vs typical",
          delta_color='inverse', border=True)
c2.metric('Restaurants ≤ 1 km', loc['competitors_1p0km'], f"{loc['competitors_1p0km'] - city['competitors_1p0km']:+.0f} vs typical",
          delta_color='inverse', border=True)
c3.metric('Restaurants ≤ 2 km', loc['competitors_2p0km'], border=True)
c4.metric('Open ≤ 1 km', loc['open_competitors_1km'], f"{loc['open_competitors_1km'] - city['open_competitors_1km']:+.0f} vs typical",
          delta_color='inverse', border=True)
c5.metric('To City Hall', f"{loc['distance_city_hall_km']:.1f} km", border=True)
st.caption('"Typical" = median over existing Philadelphia restaurants. Counts include closed restaurants, exactly as the model sees them.')

section('Competition map', f'Restaurants within {radius:g} km')
nearby = restaurants.assign(distance_km=haversine_km(lat, lon, restaurants['latitude'], restaurants['longitude']))
nearby = nearby[nearby['distance_km'] <= radius].copy()
if nearby.empty:
    st.info('No restaurants in this radius. Increase the radius in the sidebar.')
else:
    nearby['color'] = nearby['is_open'].map({1.0: [18, 140, 135, 200], 0.0: [148, 163, 184, 150]})
    nearby['status'] = nearby['is_open'].map({1.0: 'open', 0.0: 'closed'})
    map_col, list_col = st.columns([1.7, 1], gap='medium')
    with map_col:
        st.pydeck_chart(make_deck(
            layers=[pdk.Layer('ScatterplotLayer', nearby, get_position='[longitude, latitude]', get_fill_color='color',
                              get_radius='18 + review_count / 15', pickable=True, stroked=True, get_line_color=[255, 255, 255],
                              line_width_min_pixels=0.5),
                    pdk.Layer('ScatterplotLayer', pd.DataFrame({'lat': [lat], 'lon': [lon]}), get_position='[lon, lat]',
                              get_fill_color=[214, 39, 40, 240], get_radius=55, stroked=True, get_line_color=[255, 255, 255],
                              line_width_min_pixels=2)],
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=14 - radius),
            tooltip={'text': '{name}\n{stars} ★ · {review_count} reviews\n{status}'}), height=420)
        st.caption('Teal = open · grey = closed · size = review count · red = your location.')
    with list_col:
        open_share = nearby['is_open'].mean() * 100
        st.metric('Still open in this radius', f'{open_share:.0f}%', border=True)
        top = nearby.sort_values('review_count', ascending=False).head(8)
        st.markdown('**Most popular neighbours**')
        st.dataframe(top[['name', 'stars', 'review_count', 'distance_km']], hide_index=True, width='stretch', height=300,
                     column_config={'name': 'Restaurant', 'stars': st.column_config.NumberColumn('★', format='%.1f'),
                                    'review_count': st.column_config.NumberColumn('Reviews', format='%d'),
                                    'distance_km': st.column_config.NumberColumn('km', format='%.2f')})

section('What-if: your restaurant at this spot', 'The trained regression model applied to a new business')
with st.container(border=True):
    with st.form('whatif', border=False):
        f1, f2, f3 = st.columns(3, gap='large')
        with f1:
            st.markdown('**Concept**')
            tags = st.multiselect('Categories', model.tags, default=['American (New)'] if concept == 'Restaurant' else ['Coffee & Tea', 'Cafes'])
            price = st.select_slider('Price level', [1, 2, 3, 4], value=2, format_func=lambda p: '$' * p)
            stars = st.slider('Expected Yelp rating', 2.0, 5.0, 4.0, 0.5)
        with f2:
            st.markdown('**Opening hours**')
            hours = st.slider('Hours open per week', 20, 120, 70)
            days = st.slider('Days open per week', 3, 7, 7)
            late = st.slider('Days open until 11 pm or later', 0, 7, 0)
            alcohol = st.selectbox('Alcohol', ['none', 'beer & wine', 'full bar'], index=0)
        with f3:
            st.markdown('**Services**')
            attrs = {}
            for label, key, default in [('Take-out', 'RestaurantsTakeOut', True), ('Delivery', 'RestaurantsDelivery', True),
                                        ('Reservations', 'RestaurantsReservations', concept == 'Restaurant'),
                                        ('Outdoor seating', 'OutdoorSeating', False),
                                        ('Table service', 'RestaurantsTableService', concept == 'Restaurant'),
                                        ('Good for groups', 'RestaurantsGoodForGroups', True), ('Good for kids', 'GoodForKids', True),
                                        ('Happy hour', 'HappyHour', False), ('Accepts credit cards', 'BusinessAcceptsCreditCards', True)]:
                attrs[key] = st.checkbox(label, value=default)
        st.form_submit_button('Estimate popularity', type='primary', icon=':material/auto_graph:')

    estimate = model.estimate(lat=lat, lon=lon, stars=stars, price_range=price, categories=tags, attributes=attrs,
                              alcohol={'none': 0, 'beer & wine': 1, 'full bar': 2}[alcohol], days_open=days,
                              hours_per_week=hours, late_night_days=late, any_parking=1.0)
    benchmark = restaurants.loc[restaurants['competitors_1p0km'].between(loc['competitors_1p0km'] * 0.7,
                                                                        loc['competitors_1p0km'] * 1.3 + 5), 'review_count']
    left, right = st.columns([2, 1], gap='large')
    with left:
        base = alt.Chart(estimate).encode(x=alt.X('years_on_yelp:O', title='Years after opening', axis=alt.Axis(labelAngle=0)))
        chart = base.mark_line(point=True, color='#128C87').encode(
            y=alt.Y('estimated_reviews:Q', title='Estimated cumulative Yelp reviews'),
            tooltip=[alt.Tooltip('years_on_yelp', title='Years'), alt.Tooltip('estimated_reviews', title='Estimate', format='.0f')]
                    + ([alt.Tooltip('low', title='Low (10%)', format='.0f'), alt.Tooltip('high', title='High (90%)', format='.0f')]
                       if 'low' in estimate else []))
        if 'low' in estimate:
            chart = base.mark_area(opacity=.18, color='#128C87').encode(y='low:Q', y2='high:Q') + chart
        st.altair_chart(chart.properties(height=280), width='stretch')
    with right:
        row3 = estimate.loc[estimate['years_on_yelp'] == 3].iloc[0]
        st.metric('Estimated reviews after 3 years', f"{row3['estimated_reviews']:.0f}", border=True,
                  help='A popularity benchmark (Yelp reviews), not revenue')
        if 'low' in estimate:
            st.metric('Likely range (80%)', f"{row3['low']:.0f} – {row3['high']:.0f}", border=True,
                      help='Conformally calibrated: about 80% of held-out restaurants fall inside')
        if len(benchmark):
            st.metric('Typical restaurant in similarly dense areas', f'{benchmark.median():.0f} reviews', border=True)
    st.caption(f'Business age and how complete the Yelp listing is are the strongest predictors. Attributes not set here use the '
               f'catalog median ({model.typical_attribute_count} listed attributes).')

with st.expander('Model performance on the held-out test set', icon=':material/insights:'):
    metrics = load_csv(MEMBER1_DIR / 'test_metrics.csv')
    st.dataframe(metrics, hide_index=True, width='stretch', column_config={
        'MAE': st.column_config.NumberColumn(format='%.1f'), 'RMSE': st.column_config.NumberColumn(format='%.1f'),
        'R2': st.column_config.ProgressColumn('R²', min_value=0, max_value=1, format='%.2f')})
    importance = load_csv(MEMBER1_DIR / 'validation_feature_importance.csv').head(12)
    st.altair_chart(alt.Chart(importance).mark_bar(color='#128C87').encode(
        x=alt.X('MAE_increase:Q', title='Increase in validation MAE when the feature is shuffled'),
        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=260))).properties(height=320, padding={'left': 36, 'right': 8}), width='stretch')
    st.caption('Selected by validation MAE after spatial cross-validation; a 20-configuration random search and a Poisson loss '
               'did not beat it. All leakage checks passed.')

with st.expander('Explore all Philadelphia restaurants', icon=':material/table_view:'):
    all_tags = sorted({t.strip() for text in restaurants['categories'].fillna('') for t in text.split(',') if t.strip()})
    e1, e2, e3 = st.columns([2, 1, 2])
    tag = e1.selectbox('Category', ['All'] + all_tags)
    status = e2.segmented_control('Status', ['All', 'Open', 'Closed'], default='All')
    query = e3.text_input('Search by name', placeholder='e.g. Zahav')
    table = restaurants
    if tag != 'All':
        table = table[table['categories'].fillna('').map(lambda v: tag in {t.strip() for t in v.split(',')})]
    if status in ('Open', 'Closed'):
        table = table[table['is_open'].eq(1 if status == 'Open' else 0)]
    if query:
        table = table[table['name'].str.contains(query, case=False, na=False)]
    cols = ['name', 'address', 'postal_code', 'categories', 'stars', 'review_count', 'estimated_review_count',
            'competitors_1p0km', 'years_since_first_checkin', 'split']
    st.caption(f'{len(table):,} restaurants')
    st.dataframe(table[cols], hide_index=True, width='stretch', column_config={
        'stars': st.column_config.NumberColumn('★', format='%.1f'),
        'estimated_review_count': st.column_config.NumberColumn('Model estimate', format='%.0f'),
        'years_since_first_checkin': st.column_config.NumberColumn('Years on Yelp', format='%.1f'),
        'competitors_1p0km': st.column_config.NumberColumn('Competitors 1 km'), 'postal_code': st.column_config.TextColumn('ZIP')})
    st.download_button('Download CSV', table[cols].to_csv(index=False).encode('utf-8'), 'philadelphia_restaurants.csv',
                       'text/csv', icon=':material/download:')
footer()
