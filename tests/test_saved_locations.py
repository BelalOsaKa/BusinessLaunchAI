"""End-to-end test of the save-location flow: sidebar save -> finance save -> report uses it -> comparison table.

Works on a temporary copy of app/saved_locations.json, which is restored afterwards.
Run:  python -m pytest tests   (or: python tests/test_saved_locations.py)
"""
import shutil
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))


class _Page:
    """Open a view through the router (app/Home.py) after session state has been set."""

    def __init__(self, view):
        self.view, self.at = view, AppTest.from_file(str(APP / 'Home.py'), default_timeout=240)

    def __getattr__(self, name):
        return getattr(self.at, name)

    def run(self):
        self.at.run()
        self.at.switch_page(self.view).run()
        return self.at


def _page(name):
    return _Page({'3_Market_Clusters.py': 'views/clusters.py', '4_Financial_Simulator.py': 'views/finance.py',
                  '5_Launch_Report.py': 'views/report.py'}[name])


def test_save_flow():
    import common
    backup = common.SAVED_FILE.with_suffix('.bak')
    shutil.copy(common.SAVED_FILE, backup)
    try:
        # 1. Save a custom location from the sidebar.
        at = _page('3_Market_Clusters.py')
        at.session_state['preset'] = 'Custom coordinates'
        at.session_state['lat'], at.session_state['lon'] = 39.9713, -75.1340
        at = at.run()
        at.sidebar.text_input[0].input('Test Fishtown')
        at.sidebar.button[0].click().run()
        assert not at.exception, at.exception
        assert 'Test Fishtown' in common.load_saved()

        # 2. Save custom finances for it (rent changed) on the Financial Simulator page.
        at = _page('4_Financial_Simulator.py')
        at.session_state['preset'] = '★ Test Fishtown'
        at.session_state['lat'], at.session_state['lon'] = 39.9713, -75.1340
        at = at.run()
        rent = next(w for w in at.number_input if w.label.startswith('Rent'))
        rent.set_value(12345.0).run()
        next(b for b in at.button if b.label.startswith('Save these assumptions')).click().run()
        assert not at.exception, at.exception
        assert common.load_saved()['Test Fishtown']['assumptions']['rent_month'] == 12345.0

        # 3. The Launch Report uses them and compares all saved locations.
        at = _page('5_Launch_Report.py')
        at.session_state['preset'] = '★ Test Fishtown'
        at.session_state['lat'], at.session_state['lon'] = 39.9713, -75.1340
        at = at.run()
        assert not at.exception, at.exception
        assert any('using the financial assumptions you saved' in s.value.lower() for s in at.success)
        table = next(d.value for d in at.dataframe if 'location' in d.value.columns)
        assert '▶ Test Fishtown' in set(table['location']) and len(table) == len(common.load_saved())
    finally:
        shutil.move(backup, common.SAVED_FILE)


if __name__ == '__main__':
    test_save_flow()
    print('Save-location flow works end to end.')
