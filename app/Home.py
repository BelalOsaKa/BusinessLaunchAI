"""BusinessLaunch AI — entry point and navigation.  Run from the project root:  streamlit run app/Home.py"""
import streamlit as st

st.set_page_config(page_title='BusinessLaunch AI', page_icon='🍽️', layout='wide', initial_sidebar_state='expanded')

from common import check_outputs, location_picker  # noqa: E402
from ui import inject_css, score_widget  # noqa: E402

inject_css()
check_outputs()

PAGES = {
    'Start': [st.Page('views/home.py', title='Home', icon=':material/home:', default=True)],
    'Analyse the location': [
        st.Page('views/location.py', title='Location Intelligence', icon=':material/location_on:', url_path='location'),
        st.Page('views/reviews.py', title='Customer Reviews', icon=':material/reviews:', url_path='reviews'),
        st.Page('views/clusters.py', title='Market Clusters', icon=':material/hub:', url_path='clusters'),
    ],
    'Decide': [
        st.Page('views/finance.py', title='Financial Simulator', icon=':material/payments:', url_path='finance'),
        st.Page('views/report.py', title='Launch Report', icon=':material/task_alt:', url_path='report'),
    ],
    'Play': [st.Page('views/games.py', title='Beat the AI', icon=':material/sports_esports:', url_path='games')],
}
page = st.navigation(PAGES)
if page.title != 'Beat the AI':
    location_picker()
with st.sidebar:
    score_widget()
page.run()
