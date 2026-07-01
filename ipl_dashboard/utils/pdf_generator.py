import io
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def _create_styles() -> Dict[str, ParagraphStyle]:
    """Helper to initialize and customize ReportLab paragraph styles."""
    styles = getSampleStyleSheet()
    primary = colors.HexColor("#2563eb")
    text_c = colors.HexColor("#1f2937")
    
    t_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=primary, spaceAfter=15)
    st_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=12, leading=16, textColor=colors.HexColor("#4b5563"), spaceAfter=25)
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor("#1e3a8a"), spaceBefore=15, spaceAfter=10, keepWithNext=True)
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=text_c, spaceAfter=10)
    
    table_cell = ParagraphStyle('TableCell', parent=body_style, fontSize=9, leading=12, spaceAfter=0)
    table_cell_bold = ParagraphStyle('TableCellBold', parent=table_cell, fontName='Helvetica-Bold')

    return {
        "title": t_style, "subtitle": st_style, "h2": h2_style,
        "body": body_style, "table_cell": table_cell, "table_cell_bold": table_cell_bold
    }

def _build_overview_table(stats_summary: Dict[str, Any], streak_info: Dict[str, Any], best_season: Dict[str, Any], sty: Dict[str, ParagraphStyle]) -> Table:
    """Helper to compile the team performance summary table."""
    data = [
        [Paragraph("Metric", sty["table_cell_bold"]), Paragraph("Value", sty["table_cell_bold"]), Paragraph("Metric", sty["table_cell_bold"]), Paragraph("Value", sty["table_cell_bold"])],
        [Paragraph("Matches Played", sty["table_cell"]), Paragraph(str(stats_summary.get("Matches Played", 0)), sty["table_cell"]), Paragraph("Win Percentage", sty["table_cell"]), Paragraph(f"{stats_summary.get('Win Percentage (%)', 0):.1f}%", sty["table_cell"])],
        [Paragraph("Matches Won", sty["table_cell"]), Paragraph(str(stats_summary.get("Matches Won", 0)), sty["table_cell"]), Paragraph("Highest Win Streak", sty["table_cell"]), Paragraph(str(streak_info.get("Highest Winning Streak", 0)), sty["table_cell"])],
        [Paragraph("Matches Lost", sty["table_cell"]), Paragraph(str(stats_summary.get("Matches Lost", 0)), sty["table_cell"]), Paragraph("Current Form Streak", sty["table_cell"]), Paragraph(str(streak_info.get("Current Streak", 0)), sty["table_cell"])],
        [Paragraph("Best Season", sty["table_cell"]), Paragraph(f"{best_season.get('Best Season', 'N/A')} ({best_season.get('Win Rate (%)', 0):.1f}% wins)", sty["table_cell"]), Paragraph("Home Ground Record", sty["table_cell"]), Paragraph(f"{stats_summary.get('Home Wins', 0)} Wins / {stats_summary.get('Home Played', 0)} Played", sty["table_cell"])]
    ]
    t = Table(data, colWidths=[130, 130, 130, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f4f6")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t

def _build_recent_matches_table(recent_matches: List[Dict[str, Any]], sty: Dict[str, ParagraphStyle]) -> Table:
    """Helper to compile the recent match history table."""
    match_rows = [[
        Paragraph("Date", sty["table_cell_bold"]), Paragraph("Opponent", sty["table_cell_bold"]),
        Paragraph("Venue", sty["table_cell_bold"]), Paragraph("Toss Action", sty["table_cell_bold"]),
        Paragraph("Winner", sty["table_cell_bold"]), Paragraph("Result Details", sty["table_cell_bold"])
    ]]
    for m in recent_matches:
        match_rows.append([
            Paragraph(str(m.get("date", "N/A")), sty["table_cell"]), Paragraph(str(m.get("opponent", "N/A")), sty["table_cell"]),
            Paragraph(f"{m.get('venue', 'N/A')} ({m.get('city', 'N/A')})", sty["table_cell"]),
            Paragraph(f"{m.get('toss_winner', 'N/A')} ({m.get('toss_decision', 'N/A')})", sty["table_cell"]),
            Paragraph(str(m.get("winner", "N/A")), sty["table_cell"]), Paragraph(str(m.get("result", "N/A")), sty["table_cell"])
        ])
    t = Table(match_rows, colWidths=[65, 95, 130, 85, 75, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    for i in range(len(match_rows[0])):
        match_rows[0][i].style.textColor = colors.white
    return t

def generate_team_pdf(team_name: str, stats: Dict[str, Any], streaks: Dict[str, Any], best_season: Dict[str, Any], recent: List[Dict[str, Any]]) -> bytes:
    """Generates a PDF report summarizing a team's historical performance."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    sty = _create_styles()
    
    story = [
        Paragraph(f"IPL Team Report: {team_name}", sty["title"]),
        Paragraph("Generated by IPL Data Analytics Hub", sty["subtitle"]),
        Spacer(1, 10),
        Paragraph("Performance Summary", sty["h2"]),
        _build_overview_table(stats, streaks, best_season, sty),
        Spacer(1, 20),
        Paragraph("Recent Match History (Last 10 Contests)", sty["h2"]),
        _build_recent_matches_table(recent, sty)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_player_pdf(player: str, batting: Dict[str, Any], bowling: Dict[str, Any]) -> bytes:
    """Generates a PDF player profile including batting/bowling statistics."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    sty = _create_styles()
    
    story = [
        Paragraph(f"IPL Player Profile: {player}", sty["title"]),
        Paragraph("Generated by IPL Data Analytics Hub", sty["subtitle"]),
        Spacer(1, 10)
    ]
    if batting:
        story.append(Paragraph("Batting Career Statistics", sty["h2"]))
        bat_data = [[Paragraph(k, sty["table_cell_bold"]), Paragraph(str(v), sty["table_cell"])] for k, v in batting.items()]
        t = Table(bat_data, colWidths=[200, 320])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('PADDING', (0,0), (-1,-1), 6), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9fafb"))]))
        story.append(t)
        story.append(Spacer(1, 20))
    if bowling:
        story.append(Paragraph("Bowling Career Statistics", sty["h2"]))
        bowl_data = [[Paragraph(k, sty["table_cell_bold"]), Paragraph(str(v), sty["table_cell"])] for k, v in bowling.items()]
        t = Table(bowl_data, colWidths=[200, 320])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('PADDING', (0,0), (-1,-1), 6), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9fafb"))]))
        story.append(t)
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_predictor_pdf(pred_data: Dict[str, Any]) -> bytes:
    """Generates a PDF log of a win prediction scenario."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    sty = _create_styles()
    
    state_data = [[Paragraph("Parameter", sty["table_cell_bold"]), Paragraph("Value", sty["table_cell_bold"])]]
    for k in ["batting_team", "bowling_team", "city", "target_runs", "current_score", "wickets_fallen", "overs_input"]:
        state_data.append([Paragraph(k.replace('_', ' ').title(), sty["table_cell"]), Paragraph(str(pred_data[k]), sty["table_cell"])])
        
    pred_table_data = [[Paragraph("Metric", sty["table_cell_bold"]), Paragraph("Value", sty["table_cell_bold"])]]
    for k in ["model_name", "batting_win_prob", "bowling_win_prob", "confidence_level", "crr", "rrr"]:
        val = f"{pred_data[k]}%" if "prob" in k else str(pred_data[k])
        pred_table_data.append([Paragraph(k.replace('_', ' ').title(), sty["table_cell"]), Paragraph(val, sty["table_cell"])])
        
    t_state = Table(state_data, colWidths=[200, 320])
    t_state.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('PADDING', (0,0), (-1,-1), 6)]))
    
    t_pred = Table(pred_table_data, colWidths=[200, 320])
    t_pred.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ('PADDING', (0,0), (-1,-1), 6), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff"))]))
    
    story = [
        Paragraph("IPL Match Outcome Simulation", sty["title"]),
        Paragraph("Generated by ML Win Predictor Engine", sty["subtitle"]),
        Spacer(1, 10), Paragraph("Match Setup & State", sty["h2"]), t_state,
        Spacer(1, 20), Paragraph("Prediction Outcome", sty["h2"]), t_pred
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
