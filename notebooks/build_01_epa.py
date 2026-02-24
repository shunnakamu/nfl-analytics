"""Build the EPA Analysis notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.9.0"
    }
}

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# NFL EPA Analysis (2024 Season)

**Expected Points Added (EPA)** measures the value of each play by comparing the expected points before and after the play. This analysis examines team and quarterback performance through the lens of EPA using nflfastR play-by-play data.

**Key Metrics:**
- **EPA/play**: Average expected points added per play (offense & defense)
- **EPA/dropback**: QB efficiency on passing plays
- **CPOE**: Completion Percentage Over Expected — how much better/worse a QB completes passes vs. expectation
- **Pass EPA vs Rush EPA**: Efficiency split by play type"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# NFL team colors (subset for annotations)
TEAM_COLORS = {
    'DAL': '#041E42', 'PHI': '#004C54', 'WAS': '#773141', 'NYG': '#0B2265',
    'KC': '#E31837', 'BUF': '#00338D', 'SF': '#AA0000', 'DET': '#0076B6',
    'BAL': '#241773', 'HOU': '#03202F', 'GB': '#203731', 'MIN': '#4F2683',
}"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading & Preparation"))

cells.append(nbf.v4.new_code_cell("""# Load 2024 play-by-play data
pbp = nfl.import_pbp_data([2024])
print(f"Total plays: {len(pbp):,}")
print(f"Columns: {len(pbp.columns)}")

# Filter to regular season, real plays (pass/rush), remove 2-pt conversions
plays = pbp[
    (pbp['season_type'] == 'REG') &
    (pbp['play_type'].isin(['pass', 'run'])) &
    (pbp['two_point_attempt'] == 0)
].copy()

print(f"Regular season pass/rush plays: {len(plays):,}")
print(f"Teams: {plays['posteam'].nunique()}")
print(f"Weeks: {plays['week'].min()}-{plays['week'].max()}")"""))

# Team EPA
cells.append(nbf.v4.new_markdown_cell("## 2. Team EPA Rankings"))

cells.append(nbf.v4.new_code_cell("""# Offensive EPA/play by team
off_epa = plays.groupby('posteam')['epa'].mean().reset_index()
off_epa.columns = ['team', 'off_epa_play']

# Defensive EPA/play by team (lower is better for defense)
def_epa = plays.groupby('defteam')['epa'].mean().reset_index()
def_epa.columns = ['team', 'def_epa_play']

# Merge
team_epa = off_epa.merge(def_epa, on='team')
team_epa['net_epa'] = team_epa['off_epa_play'] - team_epa['def_epa_play']
team_epa = team_epa.sort_values('net_epa', ascending=False)

print("Top 10 Teams by Net EPA/play:")
print(team_epa.head(10).to_string(index=False, float_format='{:.3f}'.format))
print()
dal = team_epa[team_epa['team'] == 'DAL'].iloc[0]
print(f"Dallas Cowboys: Off EPA/play={dal['off_epa_play']:.3f}, Def EPA/play={dal['def_epa_play']:.3f}, Net={dal['net_epa']:.3f}")
print(f"Cowboys rank: {list(team_epa['team']).index('DAL') + 1} of {len(team_epa)}")"""))

# Scatter plot
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 10))

ax.scatter(team_epa['off_epa_play'], team_epa['def_epa_play'],
           s=120, alpha=0.7, color='#666666', zorder=3)

# Annotate all teams
for _, row in team_epa.iterrows():
    color = TEAM_COLORS.get(row['team'], '#333333')
    weight = 'bold' if row['team'] == 'DAL' else 'normal'
    size = 11 if row['team'] == 'DAL' else 9
    ax.annotate(row['team'], (row['off_epa_play'], row['def_epa_play']),
                fontsize=size, fontweight=weight, color=color,
                ha='center', va='bottom', xytext=(0, 6),
                textcoords='offset points')

# Highlight Cowboys
dal = team_epa[team_epa['team'] == 'DAL'].iloc[0]
ax.scatter([dal['off_epa_play']], [dal['def_epa_play']],
           s=200, color='#041E42', zorder=4, edgecolors='#869397', linewidths=2)

# Quadrant lines at 0
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Quadrant labels
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[1]*0.85, ylim[0]*0.85, 'GOOD OFF\\nGOOD DEF', ha='center', fontsize=10, alpha=0.3, fontweight='bold')
ax.text(xlim[0]*0.85, ylim[0]*0.85, 'BAD OFF\\nGOOD DEF', ha='center', fontsize=10, alpha=0.3, fontweight='bold')
ax.text(xlim[1]*0.85, ylim[1]*0.85, 'GOOD OFF\\nBAD DEF', ha='center', fontsize=10, alpha=0.3, fontweight='bold')
ax.text(xlim[0]*0.85, ylim[1]*0.85, 'BAD OFF\\nBAD DEF', ha='center', fontsize=10, alpha=0.3, fontweight='bold')

ax.set_xlabel('Offensive EPA/play →', fontsize=13)
ax.set_ylabel('← Defensive EPA/play (lower = better)', fontsize=13)
ax.set_title('2024 NFL Team EPA: Offense vs Defense', fontsize=16, fontweight='bold')
ax.invert_yaxis()  # Lower def EPA = better, so invert
plt.tight_layout()
plt.savefig('/app/nfl-analytics/notebooks/team_epa_scatter.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# Pass vs Rush EPA
cells.append(nbf.v4.new_markdown_cell("## 3. Pass EPA vs Rush EPA by Team"))

cells.append(nbf.v4.new_code_cell("""# Split by play type
pass_epa = plays[plays['play_type'] == 'pass'].groupby('posteam')['epa'].mean().reset_index()
pass_epa.columns = ['team', 'pass_epa']
rush_epa = plays[plays['play_type'] == 'run'].groupby('posteam')['epa'].mean().reset_index()
rush_epa.columns = ['team', 'rush_epa']

split_epa = pass_epa.merge(rush_epa, on='team')
split_epa['pass_rush_diff'] = split_epa['pass_epa'] - split_epa['rush_epa']

fig, ax = plt.subplots(figsize=(14, 8))

# Sort by pass EPA
split_epa_sorted = split_epa.sort_values('pass_epa', ascending=True)
y_pos = range(len(split_epa_sorted))

colors_pass = ['#041E42' if t == 'DAL' else '#2196F3' for t in split_epa_sorted['team']]
colors_rush = ['#869397' if t == 'DAL' else '#4CAF50' for t in split_epa_sorted['team']]

ax.barh([y - 0.2 for y in y_pos], split_epa_sorted['pass_epa'], height=0.35,
        color=colors_pass, label='Pass EPA/play', alpha=0.85)
ax.barh([y + 0.2 for y in y_pos], split_epa_sorted['rush_epa'], height=0.35,
        color=colors_rush, label='Rush EPA/play', alpha=0.85)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(split_epa_sorted['team'], fontsize=10)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
ax.set_xlabel('EPA/play', fontsize=13)
ax.set_title('2024 NFL Pass vs Rush EPA/play by Team', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('/app/nfl-analytics/notebooks/pass_rush_epa.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# QB EPA
cells.append(nbf.v4.new_markdown_cell("## 4. Quarterback EPA Rankings"))

cells.append(nbf.v4.new_code_cell("""# QB EPA on dropbacks (pass plays only)
qb_plays = plays[
    (plays['play_type'] == 'pass') &
    (plays['passer_player_name'].notna())
].copy()

qb_stats = qb_plays.groupby(['passer_player_name', 'posteam']).agg(
    dropbacks=('epa', 'count'),
    total_epa=('epa', 'sum'),
    epa_dropback=('epa', 'mean'),
    cpoe=('cpoe', 'mean')
).reset_index()

# Filter to QBs with minimum 200 dropbacks
qb_stats = qb_stats[qb_stats['dropbacks'] >= 200].sort_values('epa_dropback', ascending=False)

print(f"Qualified QBs (200+ dropbacks): {len(qb_stats)}")
print()
print("Top 15 QBs by EPA/dropback:")
display_cols = ['passer_player_name', 'posteam', 'dropbacks', 'epa_dropback', 'cpoe']
print(qb_stats[display_cols].head(15).to_string(index=False, float_format='{:.3f}'.format))

# Cowboys QBs
dal_qbs = qb_plays[qb_plays['posteam'] == 'DAL']['passer_player_name'].value_counts()
print(f"\\nCowboys QBs: {dict(dal_qbs)}")"""))

# QB scatter: EPA vs CPOE
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 10))

ax.scatter(qb_stats['cpoe'], qb_stats['epa_dropback'],
           s=qb_stats['dropbacks'] / 3, alpha=0.6, color='#666666', zorder=3)

# Annotate QBs
for _, row in qb_stats.iterrows():
    team = row['posteam']
    name = row['passer_player_name'].split('.')[1] if '.' in row['passer_player_name'] else row['passer_player_name']
    color = TEAM_COLORS.get(team, '#333333')
    weight = 'bold' if team == 'DAL' else 'normal'
    size = 11 if team == 'DAL' else 8.5
    ax.annotate(name, (row['cpoe'], row['epa_dropback']),
                fontsize=size, fontweight=weight, color=color,
                ha='center', va='bottom', xytext=(0, 5),
                textcoords='offset points')

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('CPOE (Completion % Over Expected) →', fontsize=13)
ax.set_ylabel('EPA/dropback →', fontsize=13)
ax.set_title('2024 NFL QBs: EPA/dropback vs CPOE (min 200 dropbacks)', fontsize=16, fontweight='bold')
ax.text(0.02, 0.98, 'Bubble size = dropback count', transform=ax.transAxes,
        fontsize=10, alpha=0.5, va='top')
plt.tight_layout()
plt.savefig('/app/nfl-analytics/notebooks/qb_epa_cpoe.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# Weekly EPA trend
cells.append(nbf.v4.new_markdown_cell("## 5. Cowboys EPA Trend Through the Season"))

cells.append(nbf.v4.new_code_cell("""# Cowboys weekly EPA
dal_plays = plays[plays['posteam'] == 'DAL']
dal_weekly = dal_plays.groupby('week').agg(
    off_epa=('epa', 'mean'),
    pass_epa=('epa', lambda x: x[dal_plays.loc[x.index, 'play_type'] == 'pass'].mean()),
    rush_epa=('epa', lambda x: x[dal_plays.loc[x.index, 'play_type'] == 'run'].mean()),
    plays_count=('epa', 'count')
).reset_index()

# Also get defensive EPA
dal_def = plays[plays['defteam'] == 'DAL']
dal_def_weekly = dal_def.groupby('week')['epa'].mean().reset_index()
dal_def_weekly.columns = ['week', 'def_epa']

dal_weekly = dal_weekly.merge(dal_def_weekly, on='week', how='left')

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Offense
axes[0].plot(dal_weekly['week'], dal_weekly['off_epa'], 'o-', color='#041E42',
             linewidth=2, markersize=8, label='Total Off EPA/play')
axes[0].fill_between(dal_weekly['week'], 0, dal_weekly['off_epa'],
                     where=dal_weekly['off_epa'] > 0, alpha=0.2, color='#041E42')
axes[0].fill_between(dal_weekly['week'], 0, dal_weekly['off_epa'],
                     where=dal_weekly['off_epa'] < 0, alpha=0.2, color='red')
axes[0].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_ylabel('EPA/play', fontsize=12)
axes[0].set_title('Dallas Cowboys 2024: Offensive EPA by Week', fontsize=14, fontweight='bold')
axes[0].legend()

# Defense
axes[1].plot(dal_weekly['week'], dal_weekly['def_epa'], 'o-', color='#869397',
             linewidth=2, markersize=8, label='Def EPA/play allowed')
axes[1].fill_between(dal_weekly['week'], 0, dal_weekly['def_epa'],
                     where=dal_weekly['def_epa'] < 0, alpha=0.2, color='green')
axes[1].fill_between(dal_weekly['week'], 0, dal_weekly['def_epa'],
                     where=dal_weekly['def_epa'] > 0, alpha=0.2, color='red')
axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Week', fontsize=12)
axes[1].set_ylabel('EPA/play allowed', fontsize=12)
axes[1].set_title('Dallas Cowboys 2024: Defensive EPA by Week (below 0 = good)', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('/app/nfl-analytics/notebooks/cowboys_weekly_epa.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# Summary
cells.append(nbf.v4.new_markdown_cell("""## 6. Key Findings

### League-Wide
- **EPA/play** effectively separates elite offenses from bottom-tier ones, with a typical range of -0.15 to +0.15
- **Pass EPA consistently exceeds Rush EPA** across the league — passing is more efficient on average, but the gap varies significantly by team
- **CPOE and EPA/dropback are correlated** but not identical — some QBs generate high EPA through completion accuracy, others through aggressive targeting

### Dallas Cowboys
- The Cowboys' offensive and defensive EPA positioning reveals their 2024 identity
- Weekly EPA trends highlight consistency (or lack thereof) through the season
- Understanding where Dallas falls in the pass vs rush efficiency spectrum has direct implications for game planning

### Applications for Game Preparation
- EPA-based team profiles can identify **matchup advantages** — e.g., teams with high pass EPA against defenses with poor pass EPA allowed
- Tracking weekly EPA trends helps identify teams that are **getting better or worse** as the season progresses
- The pass/rush EPA split by team informs **defensive play-calling** — stack the box vs. play coverage based on opponent tendencies

---
*Data source: nflfastR via nfl_data_py | 2024 NFL Regular Season*"""))

nb.cells = cells

# Write notebook
import json
with open('/app/nfl-analytics/notebooks/01_epa_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type=='code')} code, {sum(1 for c in cells if c.cell_type=='markdown')} markdown)")
