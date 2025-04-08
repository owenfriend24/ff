import pandas as pd
import ipywidgets as widgets
import seaborn as sns
import numpy as np
import ipympl
from ipywidgets import widgets, VBox, HBox
from IPython.display import display, clear_output
import matplotlib
import matplotlib.pyplot as plt
from IPython.core.display import HTML

def owner_summary():
    data = pd.read_csv('owner_summary.csv')
    
    # Extract owner names
    unique_names = sorted(data['owner'].unique().tolist())
    
    # labels for dropdown menu
    y_axis_options = {
        "Total Wins": "total_wins",
        "Total Losses": "total_losses",
        "All-Time Win Percentage": "win_percentage",
        "Total Points For": "total_pts_for",
        "Total Points Against": "total_pts_against",
        "Average Wins per Season": "avg_wins",
        "Average Losses per Season": "avg_losses",
        "Average Points For per Season": "avg_pts_for",
        "Average Points Against per Season": "avg_pts_against",
        "Average Regular Season Rank": "avg_reg_szn_rank",
        "Average Final Rank": "avg_final_rank",
        "Total Championships": "total_championships"
    }
    
    # Metrics that should be displayed as integers
    int_metrics = {"total_wins", "total_losses", "total_championships"}
    
    # Metrics w/ inverted y-axis
    inverted_metrics = {"avg_reg_szn_rank", "avg_final_rank"}
    
    custom_hex_palette = [
        "#900c3f", "#182b55", "#5f4e94", "#a291c7",
        "#82cbec", "#d94f21", "#febd2b", "#9aab4b"
    ]
    
    # assign colors to owners
    name_color_map = {name: custom_hex_palette[i % len(custom_hex_palette)] for i, name in enumerate(unique_names)}
    
    # Create name checkboxes
    name_checkboxes = {owner: widgets.Checkbox(value=True, description=str(owner)) for owner in unique_names}
    name_box = widgets.VBox([widgets.HBox(list(name_checkboxes.values())[i:i+4]) for i in range(0, len(unique_names), 4)])
    
    # Dropdown for selecting the y-axis variable 
    y_axis_dropdown = widgets.Dropdown(
        options=list(y_axis_options.keys()),  
        value="Total Wins",  
        description="Y-Axis:",
        style={'description_width': 'initial'},  
        layout=widgets.Layout(width='300px')  
    )
    
    output = widgets.Output()
    
    # Function to Update the Plot
    def update_plot(change=None):  
        with output:
            clear_output(wait=True)  
    
            # Get selected names
            selected_names = [name for name, checkbox in name_checkboxes.items() if checkbox.value]
    
            # Get selected Y-axis variable (convert display name to actual column name)
            y_axis_variable = y_axis_options[y_axis_dropdown.value]  
    
            # If no selections, show message instead of empty plot
            if not selected_names:
                display(widgets.HTML("<h3 style='color:red; font-size:16px;'>Please select at least one owner.</h3>"))
                return
    
            # Filter data based on selected owners
            filtered_data = data[data['owner'].isin(selected_names)]
    
            plt.rcParams["font.size"] = 14  
    
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
    
            # Plot bars for each selected owner
            x_positions = np.arange(len(selected_names))  
            bar_width = 0.5  
    
            for i, name in enumerate(selected_names):
                subset = filtered_data[filtered_data['owner'] == name]
                value = subset[y_axis_variable].values[0]
                color = name_color_map.get(name, "black")  # Ensure consistent color mapping
    
                # Format integer metrics properly
                label_text = f"{int(value)}" if y_axis_variable in int_metrics else f"{value:.2f}"
    
                if y_axis_variable in inverted_metrics:
                    # **Ensure bars start at the x-axis and grow upwards**
                    bar_height = 8 - value
                    ax.bar(x_positions[i], bar_height, width=bar_width, bottom=0, label=name, alpha=0.7, color=color)
                else:
                    bar_height = value
                    ax.bar(x_positions[i], value, width=bar_width, label=name, alpha=0.7, color=color)
    
                text_height = bar_height + 0.2 if bar_height > 1 else bar_height + 0.05
                ax.text(x_positions[i], text_height, label_text, ha='center', fontsize=12, fontweight='bold', color='black')
    
            ax.set_xlabel("Owner", fontsize=16)
            ax.set_ylabel(y_axis_dropdown.value, fontsize=16)
            ax.set_title(f"{y_axis_dropdown.value} Across All Seasons", fontsize=18)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(selected_names, rotation=45)
    
            if y_axis_variable in inverted_metrics:
                ax.set_ylim(0, 8) 
            else:
                max_value = filtered_data[y_axis_variable].max()
                ax.set_ylim(0, max_value * 1.1) 
    
            ax.legend(title="Owner", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14, title_fontsize=16)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    
    for checkbox in name_checkboxes.values():
        checkbox.observe(update_plot, names='value')
    y_axis_dropdown.observe(update_plot, names='value')
    
    display(widgets.HTML("<b style='font-size:16px;'>Select Owners:</b>"))
    display(name_box)
    display(y_axis_dropdown)
    
    
    display(output)
    update_plot()


def playoff_summary():
    # load playoff data
    data = pd.read_csv('playoff_metrics.csv')
    
    unique_names = sorted(data['owner'].unique().tolist())
    
    y_axis_options = {
        "Playoff Wins": "playoff_wins",
        "Playoff Points For": "playoff_pts_for",
        "Playoff Points Against": "playoff_pts_against",
        "Total Championships": "championships",
        "Last Place Finishes": "last_places",
        "Rank Differential": "rank_differential"
    }
    
    int_metrics = {"playoff_wins", "championships", "last_places"}
    
    custom_hex_palette = [
        "#900c3f", "#182b55", "#5f4e94", "#a291c7",
        "#82cbec", "#d94f21", "#febd2b", "#9aab4b"
    ]
    
    name_color_map = {name: custom_hex_palette[i % len(custom_hex_palette)] for i, name in enumerate(unique_names)}
    
    name_checkboxes = {owner: widgets.Checkbox(value=True, description=str(owner)) for owner in unique_names}
    name_box = widgets.VBox([widgets.HBox(list(name_checkboxes.values())[i:i+4]) for i in range(0, len(unique_names), 4)])
    
    y_axis_dropdown = widgets.Dropdown(
        options=list(y_axis_options.keys()),  
        value="Playoff Wins",  
        description="Y-Axis:",
        style={'description_width': 'initial'},  
        layout=widgets.Layout(width='300px')  
    )
    
    output = widgets.Output()
    
    def update_plot(change=None):  
        with output:
            clear_output(wait=True)  
    
            # Get selected names
            selected_names = [name for name, checkbox in name_checkboxes.items() if checkbox.value]
    
            # Get selected Y-axis variable
            y_axis_variable = y_axis_options[y_axis_dropdown.value]  
    
            # If no selections, show message instead of empty plot
            if not selected_names:
                display(widgets.HTML("<h3 style='color:red; font-size:16px;'>Please select at least one owner.</h3>"))
                return
    
    
            filtered_data = data[data['owner'].isin(selected_names)]
    
            plt.rcParams["font.size"] = 14  
    
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
    
            # Plot bars for each selected owner
            x_positions = np.arange(len(selected_names))  
            bar_width = 0.5  
    
            for i, name in enumerate(selected_names):
                subset = filtered_data[filtered_data['owner'] == name]
                value = subset[y_axis_variable].values[0]
                color = name_color_map.get(name, "black")
    
                # **Format integer values properly**
                label_text = f"{int(value)}" if y_axis_variable in int_metrics else f"{value:.2f}"
    
                # **Plot Rank Differential with Dynamic Range**
                if y_axis_variable == "rank_differential":
                    bar_height = value
                    ax.bar(x_positions[i], bar_height, width=bar_width, label=name, alpha=0.7, color=color)
                    
                    # **Adjust Text Placement for Rank Differential**
                    text_offset = -0.15 if value < 0 else 0.15  # Push negative values inside the bar
                    ax.text(x_positions[i], bar_height + text_offset, label_text, ha='center', fontsize=12, fontweight='bold', color='black')
    
                else:
                    bar_height = value
                    ax.bar(x_positions[i], bar_height, width=bar_width, label=name, alpha=0.7, color=color)
                    
                    # **Standard text placement for all other metrics**
                    ax.text(x_positions[i], bar_height + 0.1, label_text, ha='center', fontsize=12, fontweight='bold', color='black')
    
            ax.set_ylabel(y_axis_dropdown.value, fontsize=16)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(selected_names, rotation=45)
    
            # **Set Dynamic Range for Rank Differential**
            if y_axis_variable == "rank_differential":
                max_abs_value = max(abs(filtered_data[y_axis_variable].max()), abs(filtered_data[y_axis_variable].min()))
                buffer = 1.5 * max_abs_value
                ax.set_ylim(-buffer, buffer)  # Dynamic scaling based on data
            else:
                max_value = filtered_data[y_axis_variable].max()
                ax.set_ylim(0, max_value * 1.1)  # Add 10% space above bars
    
            # Move legend outside the plot
            ax.legend(title="Owner", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14, title_fontsize=16)
    
            ax.grid(axis="y", linestyle="--", alpha=0.7)
    
            # Adjust layout to fit legend
            plt.tight_layout()
            plt.show()
    
    
    for checkbox in name_checkboxes.values():
        checkbox.observe(update_plot, names='value')
    y_axis_dropdown.observe(update_plot, names='value')
    
    display(widgets.HTML("<b style='font-size:16px;'>Select Owners:</b>"))
    display(name_box)
    display(y_axis_dropdown)
    display(output)
    
    update_plot()



def head_to_head():
    data = pd.read_csv('head_to_head.csv')

    unique_names = sorted(data['owner'].unique().tolist())
    
    
    y_axis_options = {
        "Wins": ("owner_wins", "opp_wins"),
        "Points": ("owner_pts", "opp_pts")
    }
    
    int_metrics = {"owner_wins", "opp_wins"}
    
    custom_hex_palette = [
        "#900c3f", "#182b55", "#5f4e94", "#a291c7",
        "#82cbec", "#d94f21", "#febd2b", "#9aab4b"
    ]
    name_color_map = {name: custom_hex_palette[i % len(custom_hex_palette)] for i, name in enumerate(unique_names)}
    
    owner1_dropdown = widgets.Dropdown(
        options=unique_names,
        value=unique_names[0],
        description="Owner 1:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    owner2_dropdown = widgets.Dropdown(
        options=unique_names,
        value=unique_names[1] if len(unique_names) > 1 else unique_names[0],  # Ensure a different default
        description="Owner 2:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    y_axis_dropdown = widgets.Dropdown(
        options=list(y_axis_options.keys()),  
        value="Wins",  
        description="Y-Axis:",
        style={'description_width': 'initial'},  
        layout=widgets.Layout(width='250px')  
    )
    
    output = widgets.Output()
    
    def update_plot(change=None):  
        with output:
            clear_output(wait=True)  
    
            # Get selected owners and metric
            owner1 = owner1_dropdown.value
            owner2 = owner2_dropdown.value
            owner1_var, owner2_var = y_axis_options[y_axis_dropdown.value]
    
            # If same owner selected, show warning
            if owner1 == owner2:
                display(widgets.HTML("<h3 style='color:red; font-size:16px;'>Please select two different owners.</h3>"))
                return
    
            # **Find the matchups where Owner 1 played Owner 2**
            matchup_data = data[(data['owner'] == owner1) & (data['opponent'] == owner2)]
    
            if matchup_data.empty:
                display(widgets.HTML(f"<h3 style='color:red; font-size:16px;'>No matchups found between {owner1} and {owner2}.</h3>"))
                return
    
            # Aggregate values for total wins/points in matchups
            value1 = matchup_data[owner1_var].sum()
            value2 = matchup_data[owner2_var].sum()
    
            # Ensure proper formatting (integers vs. decimals)
            label1 = f"{int(value1)}" if owner1_var in int_metrics else f"{value1:.2f}"
            label2 = f"{int(value2)}" if owner2_var in int_metrics else f"{value2:.2f}"
    
            # Assign colors
            color1 = name_color_map.get(owner1, "black")
            color2 = name_color_map.get(owner2, "black")
    
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 6))
    
            # Plot bars side by side
            bars = ax.bar([0, 1], [value1, value2], color=[color1, color2], width=0.5)
    
            # Add text labels above bars
            for bar, label in zip(bars, [label1, label2]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, label, ha='center', fontsize=14, fontweight='bold', color='black')
    
            # Set labels and title
            ax.set_xticks([0, 1])
            ax.set_xticklabels([owner1, owner2], rotation=0, fontsize=14)
            ax.set_xlabel("Owner", fontsize=16)
            ax.set_ylabel(y_axis_dropdown.value, fontsize=16)
            ax.set_title(f"{y_axis_dropdown.value} Comparison in Matchups", fontsize=18)
    
            # Adjust Y-axis limits
            max_value = max(value1, value2)
            ax.set_ylim(0, max_value * 1.1)  # Add 10% buffer
    
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    owner1_dropdown.observe(update_plot, names='value')
    owner2_dropdown.observe(update_plot, names='value')
    y_axis_dropdown.observe(update_plot, names='value')
    
    display(widgets.HTML("<b style='font-size:16px;'>Select Owners to Compare:</b>"))
    display(widgets.HBox([owner1_dropdown, owner2_dropdown]))
    display(y_axis_dropdown)
    
    
    display(output)
    
    update_plot()


def season_by_owner():
    data = pd.read_csv('szn_summary.csv')
    
    unique_seasons = sorted(data['season'].unique().tolist())  
    unique_names = sorted(data['owner'].unique().tolist())  
    
    y_axis_options = {
        "Points For": "avg_pts_for",
        "Points Against": "avg_pts_against",
        "Total Points": "total_pts",
        "Wins": "avg_wins",
        "Losses": "avg_losses",
        "Regular Season Rank": "avg_reg_szn_rank",
        "Final Rank": "avg_final_rank",
        "Win Percentage": "win_percentage",
        "Championships Won": "total_championships"
    }
    
    int_metrics = {"avg_wins", "avg_losses", "total_championships"}
    
    inverted_metrics = {"avg_reg_szn_rank", "avg_final_rank"}
    
    custom_hex_palette = [
        "#900c3f", "#182b55", "#5f4e94", "#a291c7",
        "#82cbec", "#d94f21", "#febd2b", "#9aab4b"
    ]
    
    name_color_map = {name: custom_hex_palette[i % len(custom_hex_palette)] for i, name in enumerate(unique_names)}
    
    season_checkboxes = {season: widgets.Checkbox(value=True, description=str(season)) for season in unique_seasons}
    season_box = widgets.VBox([widgets.HBox(list(season_checkboxes.values())[i:i+4]) for i in range(0, len(unique_seasons), 4)])
    
    name_checkboxes = {owner: widgets.Checkbox(value=True, description=str(owner)) for owner in unique_names}
    name_box = widgets.VBox([widgets.HBox(list(name_checkboxes.values())[i:i+4]) for i in range(0, len(unique_names), 4)])
    
    y_axis_dropdown = widgets.Dropdown(
        options=list(y_axis_options.keys()),  
        value="Points For",  
        description="Y-Axis:",
        style={'description_width': 'initial'},  
        layout=widgets.Layout(width='250px')  
    )
    
    # allows user to specify line vs bar plotting
    plot_type_dropdown = widgets.Dropdown(
        options=['Bar', 'Line'],
        value='Bar',  
        description="Plot Type:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    output_plot = widgets.Output()
    output_table = widgets.Output()
    
    def format_table(df, title):
        """Format a Pandas DataFrame into a nicely styled HTML table with color-coded owner names."""
        styles = """
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: sans-serif;
                    font-size: 14px;
                    background-color: #f5f5f5;  /* Light Grey Background */
                }
                th {
                    background-color: #a3a3a3;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border: 1px solid #ccc;
                    text-transform: capitalize;  /* Capitalizing Column Names */
                }
                td {
                    padding: 8px;
                    border: 1px solid #ccc;
                    text-align: center;
                }
                tr:nth-child(even) {
                    background-color: #e0e0e0;
                }
                tr:hover {
                    background-color: #ddd;
                }
                .owner-box {
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 8px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                }
            </style>
        """
        
        # Format column headers to include color-coded owner boxes
        styled_columns = {
            owner: f"<span class='owner-box' style='background-color:{name_color_map.get(owner, '#000')}'>" 
                   f"{owner}</span>"
            for owner in df.columns
        }
    
        # Ensure all column names are capitalized properly
        df = df.rename(columns={col: col.capitalize() for col in df.columns})
        df = df.rename(columns=styled_columns)  # Apply styled owner columns
    
        return HTML(f"{styles}<h3 style='text-align: left; font-size: 18px;'>{title}</h3>" + df.to_html(index=True, escape=False))
    
    # Function to Update the Plot and Table
    def update_plot(change=None):
        with output_plot:
            clear_output(wait=True)
    
            # Get selected seasons and convert back to integers
            selected_seasons = [int(season) for season, checkbox in season_checkboxes.items() if checkbox.value]
    
            # Get selected names
            selected_names = [name for name, checkbox in name_checkboxes.items() if checkbox.value]
    
            # Get selected Y-axis variable
            y_axis_variable = y_axis_options[y_axis_dropdown.value]
    
            # Get selected plot type (Bar or Line)
            plot_type = plot_type_dropdown.value
    
            # If no selections, show message instead of empty plot
            if not selected_seasons or not selected_names:
                display(widgets.HTML("<h3 style='color:red; font-size:16px;'>Please select at least one season and one name.</h3>"))
                return
    
            # Filter data
            filtered_data = data[(data['season'].isin(selected_seasons)) & (data['owner'].isin(selected_names))]
    
            # Convert season to categorical values for better positioning
            season_labels = sorted(filtered_data['season'].unique())
            x_indexes = np.arange(len(season_labels)) * 1.5  
    
            bar_width = 0.15  
            num_names = len(selected_names)
            offsets = np.linspace(-bar_width * num_names / 2, bar_width * num_names / 2, num_names)
        
            plt.rcParams["font.size"] = 14  
    
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 5))
    
            for i, name in enumerate(selected_names):
                subset = filtered_data[filtered_data['owner'] == name]
                x_positions = [x_indexes[season_labels.index(s)] for s in subset['season']]
                values = subset[y_axis_variable].values
                color = name_color_map.get(name, "black")
    
                if y_axis_variable in inverted_metrics:
                    bar_values = 9 - values  
                else:
                    bar_values = values
    
                if plot_type == 'Bar':
                    ax.bar(np.array(x_positions) + offsets[i], bar_values, width=bar_width, label=name, alpha=0.7, color=color)
                else:  
                    ax.plot(x_positions, bar_values, marker='o', linestyle='-', label=name, color=color)
    
            # Set labels and title
            ax.set_xlabel("Season", fontsize=16)
            ax.set_ylabel(y_axis_dropdown.value, fontsize=16)
            ax.set_xticks(ticks=x_indexes)
            ax.set_xticklabels(season_labels, rotation=45)
    
            if y_axis_variable in inverted_metrics:
                ax.set_ylim(0, 8)
                ax.set_yticks(range(1, 9))
                ax.set_yticklabels(reversed(range(1, 9)))
            else:
                max_value = filtered_data[y_axis_variable].max()
                ax.set_ylim(0, max_value * 1.1)
    
            ax.legend(title="Owner", bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0., fontsize=12, title_fontsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
    
            plt.tight_layout()
            plt.show()
    
        with output_table:
            clear_output(wait=True)
    
            if not selected_seasons or not selected_names:
                return
    
            table_data = filtered_data.pivot(index="season", columns="owner", values=y_axis_variable)
            table_data = table_data[selected_names]  
    
            if y_axis_variable in int_metrics:
                table_data = table_data.astype("Int64")  
    
            display(format_table(table_data, f"{y_axis_dropdown.value} by Season & Owner"))
    
    
    for checkbox in season_checkboxes.values():
        checkbox.observe(update_plot, names='value')
    for checkbox in name_checkboxes.values():
        checkbox.observe(update_plot, names='value')
    
    y_axis_dropdown.observe(update_plot, names='value')
    plot_type_dropdown.observe(update_plot, names='value')
    
    display(season_box, name_box, y_axis_dropdown, plot_type_dropdown, output_plot, output_table)
    
    update_plot()


def weekly_season():
    data = pd.read_csv('full_league_history.csv')

    # Extract unique values for seasons, weeks, and names
    unique_seasons = sorted(data['season'].unique().tolist())  
    unique_weeks = sorted(data['week'].unique().tolist())  
    unique_names = sorted(data['owner'].unique().tolist())  
    
    # Define formatted labels for dropdown menu
    y_axis_options = {
        "Points For": "pts_for",
        "Points Against": "pts_against",
        "Wins": "szn_wins",
        "Losses": "szn_losses"
    }
    
    # Metrics that should be displayed as **integers**
    int_metrics = {"szn_wins", "szn_losses"}
    
    # Define a custom hex color palette (alphabetically mapped)
    custom_hex_palette = [
        "#900c3f", "#182b55", "#5f4e94", "#a291c7",
        "#82cbec", "#d94f21", "#febd2b", "#9aab4b"
    ]
    name_color_map = {name: custom_hex_palette[i % len(custom_hex_palette)] for i, name in enumerate(unique_names)}
    
    # Dropdown for selecting the Season
    season_dropdown = widgets.Dropdown(
        options=unique_seasons,
        value=unique_seasons[0],  
        description="Season:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    # Create name checkboxes
    name_checkboxes = {owner: widgets.Checkbox(value=True, description=str(owner)) for owner in unique_names}
    name_box = widgets.VBox([widgets.HBox(list(name_checkboxes.values())[i:i+4]) for i in range(0, len(name_checkboxes), 4)])
    
    # Dropdown for selecting the Y-axis variable
    y_axis_dropdown = widgets.Dropdown(
        options=list(y_axis_options.keys()),  
        value="Points For",  
        description="Y-Axis:",
        style={'description_width': 'initial'},  
        layout=widgets.Layout(width='250px')  
    )
    
    # Output widgets to contain the plot and the table
    output_plot = widgets.Output()
    output_table = widgets.Output()
    
    # Function to format and display a styled table with owner colors and light grey background
    def format_table(df, title):
        """Format a Pandas DataFrame into a nicely styled HTML table with a light grey background."""
        styles = """
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: sans-serif;
                    font-size: 14px;
                    background-color: #f5f5f5;  
                }
                th {
                    background-color: #a3a3a3;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border: 1px solid #ccc;
                    text-transform: capitalize;  /* Capitalizing Column Names */
                }
                td {
                    padding: 8px;
                    border: 1px solid #ccc;
                    text-align: center;
                }
                tr:nth-child(even) {
                    background-color: #e0e0e0;
                }
                tr:hover {
                    background-color: #ddd;
                }
                .owner-box {
                    display: inline-block;
                    padding: 4px 10px;
                    border-radius: 8px;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                }
            </style>
        """
        
        # Format column headers to include color boxes for owner names
        styled_columns = {
            owner: f"<span class='owner-box' style='background-color:{name_color_map.get(owner, '#000')}'>" 
                   f"{owner}</span>"
            for owner in df.columns
        }
    
        # Ensure all column names are capitalized properly
        df = df.rename(columns={col: col.capitalize() for col in df.columns})
        df = df.rename(columns=styled_columns)  # Apply styled owner columns
    
        return HTML(f"{styles}<h3 style='text-align: left; font-size: 18px;'>{title}</h3>" + df.to_html(index=True, escape=False))
    
    # Function to Update the Plot and Table
    def update_plot(change=None):  
        with output_plot:
            clear_output(wait=True)  
    
            # Get selected season
            selected_season = season_dropdown.value
    
            # Get selected names
            selected_names = [name for name, checkbox in name_checkboxes.items() if checkbox.value]
    
            # Get selected Y-axis variable
            y_axis_variable = y_axis_options[y_axis_dropdown.value]  
    
            # If no selections, show message instead of empty plot
            if not selected_names:
                display(widgets.HTML("<h3 style='color:red; font-size:16px;'>Please select at least one owner.</h3>"))
                return
    
            # Filter data based on the selected season
            filtered_data = data[(data['season'] == selected_season) & (data['owner'].isin(selected_names))]
    
            # Convert week to categorical values for better positioning
            week_labels = sorted(filtered_data['week'].unique())
            x_indexes = np.arange(len(week_labels))  
    
            plt.rcParams["font.size"] = 14  
    
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 5))
    
            for i, name in enumerate(selected_names):
                subset = filtered_data[filtered_data['owner'] == name].copy()
                subset = subset.sort_values(by="week")
    
                # Ensure x_positions match correct week order
                x_positions = [x_indexes[week_labels.index(w)] for w in subset['week']]
                color = name_color_map.get(name, "black")  
    
                # Plot line for each player
                ax.plot(x_positions, subset[y_axis_variable], marker='o', linestyle='-', label=name, color=color)
    
            # Set labels and title
            ax.set_xlabel("Week", fontsize=16)
            ax.set_ylabel(y_axis_dropdown.value, fontsize=16)
            ax.set_xticks(ticks=x_indexes)
            ax.set_xticklabels(week_labels)
    
            # Move legend outside the plot
            ax.legend(title="Owner", bbox_to_anchor=(1.05, 0.75), loc='upper left', borderaxespad=0., fontsize=12, title_fontsize=14)
    
            ax.grid(axis="y", linestyle="--", alpha=0.7)
    
            # Adjust layout to fit legend
            plt.tight_layout()
            plt.show()
    
        # **Update the table**
        with output_table:
            clear_output(wait=True)
    
            if not selected_names:
                return
    
            # Create formatted DataFrame for display
            table_data = filtered_data.pivot(index="week", columns="owner", values=y_axis_variable)
            table_data = table_data[selected_names]  
    
            # Format integer values correctly
            if y_axis_variable in int_metrics:
                table_data = table_data.astype("Int64")  
    
            # Display styled table with light grey background & capitalized headers
            display(format_table(table_data, f"{y_axis_dropdown.value} by Week & Owner"))
    
    # Attach observers for automatic updates
    season_dropdown.observe(update_plot, names='value')
    for checkbox in name_checkboxes.values():
        checkbox.observe(update_plot, names='value')
    y_axis_dropdown.observe(update_plot, names='value')
    
    # Display UI elements
    display(widgets.HTML("<b style='font-size:16px;'>Select Season:</b>"))
    display(season_dropdown)
    display(widgets.HTML("<b style='font-size:16px;'>Select Owners:</b>"))
    display(name_box)
    display(y_axis_dropdown)
    
    # Display the output widgets
    display(output_plot)
    display(output_table)
    
    # Initial plot
    update_plot()

