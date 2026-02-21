import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# Import the teams dictionary from Table.py
from Table import teams

def calculate_table_stats(csv_file='A:\ML\La Liga Match Predictor\Predicted_Fixtures_Results.csv'):
    """Calculate full table statistics from the fixtures CSV."""
    df = pd.read_csv(csv_file)
    
    # Initialize stats dictionary
    stats = {}
    for team in teams.keys():
        stats[team] = {
            'points': 0,
            'played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0
        }
    
    # Process each match
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = int(row['PredHomeGoals'])
        away_goals = int(row['PredAwayGoals'])
        result = row['Result']
        
        # Update played matches
        stats[home_team]['played'] += 1
        stats[away_team]['played'] += 1
        
        # Update goals
        stats[home_team]['goals_for'] += home_goals
        stats[home_team]['goals_against'] += away_goals
        stats[away_team]['goals_for'] += away_goals
        stats[away_team]['goals_against'] += home_goals
        
        # Update results and points
        if result == 'H':
            stats[home_team]['wins'] += 1
            stats[home_team]['points'] += 3
            stats[away_team]['losses'] += 1
        elif result == 'A':
            stats[away_team]['wins'] += 1
            stats[away_team]['points'] += 3
            stats[home_team]['losses'] += 1
        else:  # Draw
            stats[home_team]['draws'] += 1
            stats[home_team]['points'] += 1
            stats[away_team]['draws'] += 1
            stats[away_team]['points'] += 1
    
    return stats

def resize_image(image_path, target_size=(50, 50)):
    """Resize image to consistent size while maintaining aspect ratio."""
    try:
        img = Image.open(image_path)
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Resize maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        # Create a square canvas
        canvas = Image.new('RGBA', target_size, (255, 255, 255, 0))
        # Center the image on the canvas
        offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
        canvas.paste(img, offset)
        return canvas
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def create_table_display():
    """Create and display the La Liga table."""
    # Calculate statistics
    stats = calculate_table_stats()
    
    # Create DataFrame for the table
    table_data = []
    for team, team_stats in stats.items():
        goal_diff = team_stats['goals_for'] - team_stats['goals_against']
        table_data.append({
            'Team': team,
            'Points': team_stats['points'],
            'Wins': team_stats['wins'],
            'Draws': team_stats['draws'],
            'Losses': team_stats['losses'],
            'GD': goal_diff
        })
    
    # Sort by points (descending), then by goal difference
    df_table = pd.DataFrame(table_data)
    df_table = df_table.sort_values(by=['Points', 'GD'], ascending=[False, False])
    df_table.reset_index(drop=True, inplace=True)
    df_table.index = df_table.index + 1  # Position starts at 1
    
    # Create figure with white background
    fig = plt.figure(figsize=(14, 18), facecolor='white')
    
    # Add La Liga logo at the top
    try:
        laliga_logo = Image.open('A:\ML\La Liga Match Predictor\images\laliga.png')
        if laliga_logo.mode != 'RGBA':
            laliga_logo = laliga_logo.convert('RGBA')
        logo_ax = fig.add_axes([0.3, 0.94, 0.4, 0.05])
        logo_ax.imshow(laliga_logo)
        logo_ax.axis('off')
    except Exception as e:
        print(f"Warning: Could not load laliga.png - {e}")
    
    # Table positioning
    table_ax = fig.add_axes([0.08, 0.05, 0.84, 0.87])
    table_ax.axis('off')
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(0, 1)
    
    # Define column widths and colors
    col_widths = [0.07, 0.09, 0.32, 0.12, 0.1, 0.1, 0.1, 0.1]
    cell_height = 0.042
    
    # Color scheme
    header_bg = '#2C3E50'  # Dark blue-grey
    header_text = 'white'
    even_row_bg = '#ECF0F1'  # Light grey
    odd_row_bg = 'white'
    border_color = '#BDC3C7'  # Grey border
    
    # Headers
    headers = ['Pos', 'Logo', 'Team', 'Pts', 'W', 'D', 'L', 'GD']
    
    y_position = 0.97
    
     # Draw header row
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        x_position = sum(col_widths[:i])
        rect = plt.Rectangle((x_position, y_position), width, cell_height, 
                            facecolor=header_bg, edgecolor=border_color, linewidth=1.5)
        table_ax.add_patch(rect)
        # Center align all header text
        table_ax.text(x_position + width/2, y_position + cell_height/2, header,
                     ha='center', va='center', fontsize=12, fontweight='bold', 
                     color=header_text, family='sans-serif')
    
    y_position -= cell_height
    
    # Draw data rows
    for pos, row in df_table.iterrows():
        # Alternate row colors
        row_color = even_row_bg if pos % 2 == 0 else odd_row_bg
        
        # Special coloring for top 4 (Champions League) and bottom 3 (Relegation)
        if pos <= 4:
            position_color = '#3498DB'  # Blue for CL spots
        elif pos >= 18:
            position_color = '#E74C3C'  # Red for relegation
        else:
            position_color = '#95A5A6'  # Grey for mid-table
        
        # Draw cells
        for i, width in enumerate(col_widths):
            x_position = sum(col_widths[:i])
            rect = plt.Rectangle((x_position, y_position), width, cell_height,
                                facecolor=row_color, edgecolor=border_color, linewidth=0.8)
            table_ax.add_patch(rect)
        
        # Position with colored indicator
        pos_x = sum(col_widths[:0])
        pos_indicator = plt.Rectangle((pos_x, y_position), 0.01, cell_height,
                                     facecolor=position_color, edgecolor='none')
        table_ax.add_patch(pos_indicator)
        table_ax.text(pos_x + col_widths[0]/2, y_position + cell_height/2,
                     str(pos), ha='center', va='center', fontsize=11, 
                     fontweight='bold', family='sans-serif')
        
        # Team logo with consistent sizing
        logo_img = resize_image(f"A:\ML\La Liga Match Predictor\svg\png\{row['Team']}.png", target_size=(50, 50))
        if logo_img is not None:
            imagebox = OffsetImage(logo_img, zoom=0.5)
            ab = AnnotationBbox(imagebox, 
                              (sum(col_widths[:1]) + col_widths[1]/2, y_position + cell_height/2),
                              frameon=False, box_alignment=(0.5, 0.5))
            table_ax.add_artist(ab)
        
        # Team name
        table_ax.text(sum(col_widths[:2]) + 0.01, y_position + cell_height/2,
                     row['Team'], ha='left', va='center', fontsize=11, family='sans-serif')
        
        # Points (bold)
        table_ax.text(sum(col_widths[:3]) + col_widths[3]/2, y_position + cell_height/2,
                     str(row['Points']), ha='center', va='center', fontsize=11, 
                     fontweight='bold', family='sans-serif')
        
        # Wins
        table_ax.text(sum(col_widths[:4]) + col_widths[4]/2, y_position + cell_height/2,
                     str(row['Wins']), ha='center', va='center', fontsize=10, family='sans-serif')
        
        # Draws
        table_ax.text(sum(col_widths[:5]) + col_widths[5]/2, y_position + cell_height/2,
                     str(row['Draws']), ha='center', va='center', fontsize=10, family='sans-serif')
        
        # Losses
        table_ax.text(sum(col_widths[:6]) + col_widths[6]/2, y_position + cell_height/2,
                     str(row['Losses']), ha='center', va='center', fontsize=10, family='sans-serif')
        
        # Goal difference (colored)
        gd = row['GD']
        gd_text = f"+{gd}" if gd > 0 else str(gd)
        gd_color = '#27AE60' if gd > 0 else '#E74C3C' if gd < 0 else '#34495E'
        table_ax.text(sum(col_widths[:7]) + col_widths[7]/2, y_position + cell_height/2,
                     gd_text, ha='center', va='center', fontsize=10, 
                     fontweight='bold', color=gd_color, family='sans-serif')
        
        y_position -= cell_height
    
    plt.tight_layout()
    plt.savefig('laliga_table.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("\nTable saved as 'laliga_table.png'")
    print("\nFinal Standings:")
    print(df_table.to_string(index=True))

if __name__ == "__main__":
    create_table_display()