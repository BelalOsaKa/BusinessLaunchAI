"""Smoke tests: every page renders through the real navigation (app/Home.py) for several locations and both concepts.

Run from the project root:  python -m pytest tests   (or: python tests/test_app_pages.py)
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))
VIEWS = ['views/home.py', 'views/location.py', 'views/reviews.py', 'views/clusters.py', 'views/finance.py',
         'views/report.py', 'views/games.py']
CASES = [('Center City – City Hall', 'Restaurant'), ('East Passyunk', 'Café'), ('Northeast – Frankford Ave', 'Restaurant')]


def run_view(view, preset, concept):
    from common import PRESETS
    at = AppTest.from_file(str(APP / 'Home.py'), default_timeout=240)
    at.session_state['preset'] = preset
    at.session_state['lat'], at.session_state['lon'] = PRESETS[preset]
    at.session_state['concept'] = concept
    at.run()
    if view != 'views/home.py':
        at.switch_page(view).run()
    return at


def test_all_pages():
    failures = []
    for view in VIEWS:
        for preset, concept in CASES:
            at = run_view(view, preset, concept)
            for problem in list(at.exception) + list(at.error):
                failures.append(f'{view} / {preset} / {concept}: {problem.value}')
    assert not failures, '\n'.join(failures)


def test_games_are_playable():
    """Play one round of each game by clicking its answer buttons."""
    at = run_view('views/games.py', 'Center City – City Hall', 'Restaurant')
    for game, prefix in [('🔥 Higher or Lower', 'A has more'), ('🕵️ Complaint Detective', None), ('📍 Pick the Spot', 'Open in A')]:
        at.segmented_control(key='game_choice').set_value(game).run()
        assert not at.exception, at.exception
        buttons = [b for b in at.button if (prefix and b.label.startswith(prefix)) or (prefix is None and b.key and b.key.startswith('det_'))]
        buttons[0].click().run()
        assert not at.exception, at.exception
    assert at.session_state['hl_stats']['rounds'] == 1
    assert at.session_state['det_stats']['rounds'] == 1 and at.session_state['spot_stats']['rounds'] == 1


if __name__ == '__main__':
    test_all_pages()
    test_games_are_playable()
    print(f'All {len(VIEWS)} pages rendered for {len(CASES)} locations; all three games are playable.')
