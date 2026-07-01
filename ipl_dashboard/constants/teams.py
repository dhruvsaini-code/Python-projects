from typing import Dict

# Standardizing team names (modernizing historical ones, correcting spellings)
TEAM_NAME_MAP: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}

# Brand colors for standard teams (primary colors for clean visualizations)
TEAM_COLORS: Dict[str, str] = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#FDB913",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#134B8C",
    "Sunrisers Hyderabad": "#FF822E",
    "Punjab Kings": "#DD1F26",
    "Rajasthan Royals": "#EA1B85",
    "Gujarat Titans": "#1B2544",
    "Lucknow Super Giants": "#0057E7",
    "Rising Pune Supergiant": "#D11D5B",
    "Kochi Tuskers Kerala": "#E66E19",
    "Pune Warriors": "#2F9E97",
    "Gujarat Lions": "#E65A28",
    "Unknown": "#7f8c8d",
    "No Result": "#95a5a6"
}

def get_team_color(team_name: str) -> str:
    """Returns the brand color for a team, fallback to standard neutral gray if unknown."""
    return TEAM_COLORS.get(team_name, "#7f8c8d")
