"""Shared visual theme for the presentation and its charts -- a calm, muted
dark slate-blue palette, used both by matplotlib (chart regeneration) and
python-pptx (slide building) so the two stay visually consistent.
"""

BG = "#12161C"          # near-black, neutral cool slate
BG_ALT = "#181E26"      # slightly lighter panel background
CARD_BG = "#1E2530"     # card fill
CARD_BORDER = "#2E3846"

PRIMARY = "#4A7FAE"     # primary accent -- calm steel blue
PRIMARY_DARK = "#31577A"
AMBER = "#C99A4A"       # muted amber -- warnings / neutral highlight
GOLD = "#D4B876"        # soft muted gold
GREEN = "#5FA383"       # muted sage green -- positive results
GRAY_TEXT = "#A9B4C0"   # muted cool-gray body text on dark bg
WHITE = "#F2F4F7"
LINE_GRID = "#2E3846"

MODEL_COLORS = {
    "random_targeting": "#7C8894",
    "response_catboost": "#C4685F",
    "t_learner_catboost": "#C99A4A",
    "x_learner_catboost": "#4A7FAE",
    "transformer_two_head": "#5FA383",
}

MODEL_LABELS = {
    "random_targeting": "Random",
    "response_catboost": "Response CatBoost",
    "t_learner_catboost": "T-Learner",
    "x_learner_catboost": "X-Learner",
    "transformer_two_head": "Transformer",
}

FONT = "Calibri"
