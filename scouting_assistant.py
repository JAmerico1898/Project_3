import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import plotly.express as px
import io
import matplotlib.pyplot as plt
from soccerplots.radar_chart import Radar
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
from dotenv import load_dotenv
import streamlit as st
import openai

# Load environment variables from .env file
#load_dotenv()

# Get the API key
#api_key = os.getenv("GEMINI_API_KEY")

#CABEÇALHO DO FORM
st.markdown("<h1 style='text-align: center;'>Scouting Assistant</h1>", unsafe_allow_html=True)
#st.markdown("<h6 style='text-align: center;'>app by @JAmerico1898</h6>", unsafe_allow_html=True)
st.markdown("---")

#with st.sidebar:

    #jogadores = df1["Atleta"]
#    choose = option_menu("Galeria de Apps", ["Individual Analysis"],
#                         icons=['graph-up-arrow'],
#                         menu_icon="universal-access", default_index=0, 
#                         styles={
#                         "container": {"padding": "5!important", "background-color": "#fafafa"},
#                         "icon": {"color": "orange", "font-size": "25px"},
#                         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
#                         "nav-link-selected": {"background-color": "#02ab21"},    
#                         }
#                         )


df13 = pd.read_csv("base_jogadores_scouting_assistant_2024.csv")
df16 = pd.read_csv("atributos.csv")
prospectos = df13["player_name"]
clubes = df13["Team within selected timeframe"]
posições = df13["role_2"]
atributos_lateral = ["involvement", "active_defence", "intelligent_defence", 
                     "territorial_dominance", "chance_prevention", "defensive_heading", 
                     "aerial_threat", "composure", "progression"]
atributos_lateral_2 = ["involvement", "active_defence", "intelligent_defence", 
                     "territorial_dominance", "chance_prevention", "defensive_heading", 
                     "aerial_threat", "composure", "progression"]
atributos_zagueiro = ["involvement", "active_defence", "intelligent_defence", 
                     "territorial_dominance", "chance_prevention","progression",
                     "passing_quality", "run_quality", "providing_teammates"]
atributos_meio_campo = ["involvement", "active_defence", "intelligent_defence",
                        "progression", "passing_quality", "effectiveness", 
                        "providing_teammates", "box_threat"]
atributos_meio_campo = ["involvement", "active_defence", "intelligent_defence",
                        "progression", "passing_quality", "effectiveness", 
                        "providing_teammates", "box_threat"]
atributos_extremo = ["involvement", "pressing", "passing_quality", "run_quality",
                     "dribbling", "effectiveness", "providing_teammates", "box_threat",
                     "finishing"]
atributos_extremo_2 = ["involvement", "pressing", "passing_quality", "run_quality",
                     "dribbling", "effectiveness", "providing_teammates", "box_threat",
                     "finishing"]
atributos_atacante = ["involvement", "pressing", "aerial_threat", "hold-up_play",
                      "run_quality", "providing_teammates", "finishing", "poaching"]
atributos_atacante_2 = ["involvement", "pressing", "aerial_threat", "hold-up_play",
                      "run_quality", "providing_teammates", "finishing", "poaching"]


#if choose == "Individual Analysis":

jogadores = st.selectbox("Choose the player", options=prospectos, index = None)
if jogadores:
    #Determinar Posição
    df14 = df13.loc[(df13['player_name']==jogadores)]
    posições = df14['role_2'].unique()
    posição = st.selectbox("Choose the position", options=posições)


##########################################################################################################################
##########################################################################################################################

    if posição == ("STRIKER"):
        #Plotar Primeiro Gráfico - Radar de Percentis do Jogador na liga:
        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Percentiles)</h3>", unsafe_allow_html=True)
        Lateral_Charts = pd.read_csv('variable_df_adj_final_per.csv')
        Lateral_Charts_1 = Lateral_Charts[(Lateral_Charts['player_name']==jogadores)&(Lateral_Charts['role_2']==posição)]
        columns_to_rename = {
            col: col.replace('_per', '') for col in Lateral_Charts.columns if '_per' in col
        }
        # Renaming the columns in the DataFrame
        Lateral_Charts_1.rename(columns=columns_to_rename, inplace=True)
        #Collecting data to plot
        metrics = Lateral_Charts_1.iloc[:, np.r_[27, 30, 17, 25, 33, 32, 24, 29]].reset_index(drop=True)
        metrics_list = metrics.iloc[0].tolist()
        #Collecting clube
        clube = Lateral_Charts_1.iat[0, 4]
        
        ## parameter names
        params = metrics.columns.tolist()

        ## range values
        ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

        ## parameter value
        values = metrics_list

        ## title values
        title = dict(
            title_name=jogadores,
            title_color = 'blue',
            subtitle_name= (posição),
            subtitle_color='#344D94',
            title_name_2=clube,
            title_color_2 = 'blue',
            subtitle_name_2='2023',
            subtitle_color_2='#344D94',
            title_fontsize=20,
            subtitle_fontsize=18,
        )            

        ## endnote 
        endnote = ""

        ## instantiate object
        radar = Radar()

        ## instantiate object -- changing fontsize
        radar=Radar(fontfamily='Cursive', range_fontsize=13)
        radar=Radar(fontfamily='Cursive', label_fontsize=15)

        ## plot radar -- filename and dpi
        fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=[('#B6282F', 0.65), ('#344D94', 0.65)], 
                                title=title, endnote=endnote, dpi=600)
        st.pyplot(fig)

        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        #Plotar Segundo Gráfico - Dispersão dos jogadores da mesma posição na liga em eixo único:

        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)


        # Dynamically create the HTML string with the 'jogadores' variable
        title_html = f"<h3 style='text-align: center; font-weight: bold; color: blue;'>{jogadores}</h3>"

        # Use the dynamically created HTML string in st.markdown
        st.markdown(title_html, unsafe_allow_html=True)

        #st.markdown("<h3 style='text-align: center;'>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)
        # Collecting data
        Lateral_Charts_2 = pd.read_csv('variable_df_adj_final_z.csv')
        Lateral_Charts_2 = Lateral_Charts_2[(Lateral_Charts['role_2']==posição)]

        #Collecting data to plot
        metrics = Lateral_Charts_2.iloc[:, np.r_[26, 29, 16, 24, 32, 31, 23, 28]].reset_index(drop=True)
        metrics_involvement = metrics.iloc[:, 0].tolist()
        metrics_pressing = metrics.iloc[:, 1].tolist()
        metrics_aerial_threat = metrics.iloc[:, 2].tolist()
        metrics_hold_up_play = metrics.iloc[:, 3].tolist()
        metrics_run_quality = metrics.iloc[:, 4].tolist()
        metrics_providing_teammates = metrics.iloc[:, 5].tolist()
        metrics_finishing = metrics.iloc[:, 6].tolist()
        metrics_poaching = metrics.iloc[:, 7].tolist()
        metrics_y = [0] * len(metrics_involvement)

        # The specific data point you want to highlight
        highlight = Lateral_Charts_2[(Lateral_Charts_2['player_name']==jogadores)]
        highlight = highlight.iloc[:, np.r_[26, 29, 16, 24, 32, 31, 23, 28]].reset_index(drop=True)
        highlight_involvement = highlight.iloc[:, 0].tolist()
        highlight_pressing = highlight.iloc[:, 1].tolist()
        highlight_aerial_threat = highlight.iloc[:, 2].tolist()
        highlight_hold_up_play = highlight.iloc[:, 3].tolist()
        highlight_run_quality = highlight.iloc[:, 4].tolist()
        highlight_providing_teammates = highlight.iloc[:, 5].tolist()
        highlight_finishing = highlight.iloc[:, 6].tolist()
        highlight_poaching = highlight.iloc[:, 7].tolist()
        highlight_y = 0

        # Computing the selected player specific values
        highlight_involvement_value = pd.DataFrame(highlight_involvement).reset_index(drop=True)
        highlight_pressing_value = pd.DataFrame(highlight_pressing).reset_index(drop=True)
        highlight_aerial_threat_value = pd.DataFrame(highlight_aerial_threat).reset_index(drop=True)
        highlight_hold_up_play_value = pd.DataFrame(highlight_hold_up_play).reset_index(drop=True)
        highlight_run_quality_value = pd.DataFrame(highlight_run_quality).reset_index(drop=True)
        highlight_providing_teammates_value = pd.DataFrame(highlight_providing_teammates).reset_index(drop=True)
        highlight_finishing_value = pd.DataFrame(highlight_finishing).reset_index(drop=True)
        highlight_poaching_value = pd.DataFrame(highlight_poaching).reset_index(drop=True)

        highlight_involvement_value = highlight_involvement_value.iat[0,0]
        highlight_pressing_value = highlight_pressing_value.iat[0,0]
        highlight_aerial_threat_value = highlight_aerial_threat_value.iat[0,0]
        highlight_hold_up_play_value = highlight_hold_up_play_value.iat[0,0]
        highlight_run_quality_value = highlight_run_quality_value.iat[0,0]
        highlight_providing_teammates_value = highlight_providing_teammates_value.iat[0,0]
        highlight_finishing_value = highlight_finishing_value.iat[0,0]
        highlight_poaching_value = highlight_poaching_value.iat[0,0]

        # Computing the min and max value across all lists using a generator expression
        min_value = min(min(lst) for lst in [metrics_involvement, metrics_pressing, metrics_aerial_threat, 
                                            metrics_hold_up_play, metrics_run_quality, 
                                            metrics_providing_teammates, metrics_finishing, metrics_poaching])
        min_value = min_value - 0.1
        max_value = max(max(lst) for lst in [metrics_involvement, metrics_pressing, metrics_aerial_threat, 
                                            metrics_hold_up_play, metrics_run_quality,
                                            metrics_providing_teammates, metrics_finishing, metrics_poaching])
        max_value = max_value + 0.1

        # Create two subplots vertically aligned with separate x-axes
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1)
        ax.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        #Collecting Additional Information
        # Load the saved DataFrame from "Lateral_ranking.csv"
        lateral_ranking_df = pd.read_csv("variable_df_adj_final_rank.csv")

        # Building the Extended Title"
        rows_count = lateral_ranking_df[lateral_ranking_df['role_2'] == "STRIKER"].shape[0]
        involvement_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'involvement'].values
        involvement_ranking_value = involvement_ranking_value[0].astype(int)
        output_str = f"({involvement_ranking_value}/{rows_count})"
        full_title_involvement = f"Involvement {output_str} {highlight_involvement_value}"

        # Building the Extended Title"
        pressing_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'pressing'].values
        pressing_ranking_value = pressing_ranking_value[0].astype(int)
        output_str = f"({pressing_ranking_value}/{rows_count})"
        full_title_pressing = f"Pressing {output_str} {highlight_pressing_value}"

        # Building the Extended Title"
        aerial_threat_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'aerial_threat'].values
        aerial_threat_ranking_value = aerial_threat_ranking_value[0].astype(int)
        output_str = f"({aerial_threat_ranking_value}/{rows_count})"
        full_title_aerial_threat = f"Aerial threat {output_str} {highlight_aerial_threat_value}"

        # Building the Extended Title"
        hold_up_play_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'hold-up_play'].values
        hold_up_play_ranking_value = hold_up_play_ranking_value[0].astype(int)
        output_str = f"({hold_up_play_ranking_value}/{rows_count})"
        full_title_hold_up_play = f"hold-up play {output_str} {highlight_hold_up_play_value}"
        
        # Building the Extended Title"
        run_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'run_quality'].values
        run_quality_ranking_value = run_quality_ranking_value[0].astype(int)
        output_str = f"({run_quality_ranking_value}/{rows_count})"
        full_title_run_quality = f"Run quality {output_str} {highlight_run_quality_value}"

        # Building the Extended Title"
        providing_teammates_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'providing_teammates'].values
        providing_teammates_ranking_value = providing_teammates_ranking_value[0].astype(int)
        output_str = f"({providing_teammates_ranking_value}/{rows_count})"
        full_title_providing_teammates = f"Providing teammates {output_str} {highlight_providing_teammates_value}"

        # Building the Extended Title"
        finishing_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'finishing'].values
        finishing_ranking_value = finishing_ranking_value[0].astype(int)
        output_str = f"({finishing_ranking_value}/{rows_count})"
        full_title_finishing = f"Finishing {output_str} {highlight_finishing_value}"

        # Building the Extended Title"
        poaching_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "STRIKER"), 'poaching'].values
        poaching_ranking_value = poaching_ranking_value[0].astype(int)
        output_str = f"({poaching_ranking_value}/{rows_count})"
        full_title_poaching = f"Poaching {output_str} {highlight_poaching_value}"

        # Plot the first scatter plot in the first subplot
        ax1.scatter(metrics_involvement, metrics_y, color='deepskyblue')
        ax1.scatter(highlight_involvement, highlight_y, color='blue', s=60)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title(full_title_involvement, fontsize=9, fontweight='bold')
        ax1.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax2.scatter(metrics_pressing, metrics_y, color='deepskyblue')
        ax2.scatter(highlight_pressing, highlight_y, color='blue', s=60)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(full_title_pressing, fontsize=9, fontweight='bold')
        ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax3.scatter(metrics_aerial_threat, metrics_y, color='deepskyblue')
        ax3.scatter(highlight_aerial_threat, highlight_y, color='blue', s=60)            
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(full_title_aerial_threat, fontsize=9, fontweight='bold')
        ax3.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax3.xaxis.set_major_locator(ticker.NullLocator())
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax4.scatter(metrics_hold_up_play, metrics_y, color='deepskyblue')
        ax4.scatter(highlight_hold_up_play, highlight_y, color='blue', s=60)            
        ax4.get_yaxis().set_visible(False)
        ax4.set_title(full_title_hold_up_play, fontsize=9, fontweight='bold')
        ax4.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax4.xaxis.set_major_locator(ticker.NullLocator())
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax5.scatter(metrics_run_quality, metrics_y, color='deepskyblue')
        ax5.scatter(highlight_run_quality, highlight_y, color='blue', s=60)
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(full_title_run_quality, fontsize=9, fontweight='bold')
        ax5.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax5.xaxis.set_major_locator(ticker.NullLocator())
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax6.scatter(metrics_providing_teammates, metrics_y, color='deepskyblue')
        ax6.scatter(highlight_providing_teammates, highlight_y, color='blue', s=60)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(full_title_providing_teammates, fontsize=9, fontweight='bold')
        ax6.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax6.xaxis.set_major_locator(ticker.NullLocator())
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax7.scatter(metrics_finishing, metrics_y, color='deepskyblue', label='Other players in the league')
        ax7.scatter(highlight_finishing, highlight_y, color='blue', s=60, label=jogadores)
        ax7.get_yaxis().set_visible(False)
        ax7.set_title(full_title_finishing, fontsize=9, fontweight='bold')
        ax7.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax7.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax7.spines['left'].set_visible(False)
        ax7.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax8.scatter(metrics_poaching, metrics_y, color='deepskyblue', label='Other players in the league')
        ax8.scatter(highlight_poaching, highlight_y, color='blue', s=60, label=jogadores)
        ax8.get_yaxis().set_visible(False)
        ax8.set_title(full_title_poaching, fontsize=9, fontweight='bold')
        ax8.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax8.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['left'].set_visible(False)
        ax8.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        ax8.legend(loc='right', bbox_to_anchor=(0.2, -2.5), fontsize="6", frameon=False)
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()

        st.pyplot(fig)
        
#####################################################################################################################            
#####################################################################################################################            

        def describe(thresholds, words, value):
            """
            Converts a z-score to a descriptive word based on predefined thresholds.
            
            Args:
                thresholds (list): Ordered list of z-score thresholds
                words (list): Corresponding descriptive words
                value (float): Z-score to categorize
            
            Returns:
                str: Descriptive word for the z-score
            """
            # Ensure value is converted to float and handle potential errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                return "unavailable"
            
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return words[i]
            return words[-1]

        def describe_level(value):
            """
            Describes a player's metric performance level.
            
            Args:
                value (float): Z-score of a metric
            
            Returns:
                str: Descriptive performance level
            """
            thresholds = [1.5, 1, 0.5, -0.5, -1]
            words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
            return describe(thresholds, words, value)

        def load_player_qa_context(player_name):
            """
            Load Q&A context for a specific player from Excel.
            
            Args:
                player_name (str): Name of the player
            
            Returns:
                str: Formatted context from Q&A pairs
            """
            try:
                # Read the Excel file
                qa_df = pd.read_excel('striker_players_user_assistant.xlsx')
                
                # Filter for the specific player
                player_qa = qa_df[qa_df['user'] == jogadores]
                
                # If no rows found, return empty string
                if player_qa.empty:
                    return ""
                
                # Format Q&A pairs into a context string
                qa_context = []
                for _, row in player_qa.iterrows():
                    if pd.notna(row['user']) and pd.notna(row['assistant']):
                        qa_context.append(f"Q: {row['user']}\nA: {row['assistant']}")
                
                return "\n\n".join(qa_context)
            
            except Exception as e:
                st.error(f"Error loading Q&A context: {e}")
                return ""

        def generate_player_description(player_data):
            """
            Generate a descriptive summary of a player's performance.
            
            Args:
                player_data (pd.Series): Row containing player's z-scores
            
            Returns:
                str: Descriptive narrative about the player
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
            # Prepare descriptive metrics (only for numeric columns)
            metrics_description = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])

    
            # Load Q&A context for the player
            qa_context = load_player_qa_context(player_data['player_name'])
 
            
            prompt = (
                f"Please use the statistical description to give a concise, 4 sentence summary of {player_data['player_name']}'s playing style, strengths and weaknesses. "
                f"Statistical Context: {metrics_description}. "
                "The first sentence should use varied language to give an overview of the player. "
                "The second sentence should describe the player's specific strengths based on the metrics. "
                "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
                "Finally, summarise exactly how the player compares to others in the same position. "
            )
            
            response = model.generate_content(prompt)
            return response.text

        def initialize_chat_history(player_name, initial_description):
            """
            Initialize the chat history with the player description.
            
            Args:
                player_name (str): Name of the selected player
                initial_description (str): Initial AI-generated player description
            """
            # Reset chat history
            st.session_state.messages = [
                {"role": "assistant", "content": f"Player Analysis for {player_name}:"},
                {"role": "assistant", "content": initial_description}
            ]

        def generate_chat_response(prompt, player_name, player_data):
            """
            Generate a response from Gemini based on the chat prompt.
            
            Args:
                prompt (str): User's chat input
                player_name (str): Name of the selected player
                player_data (pd.Series): Player's statistical data
            
            Returns:
                str: AI-generated response
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')

    
            # Load Q&A context for the player
            qa_context = load_player_qa_context(player_name)
    
            
            # Prepare additional context from player metrics
            metrics_context = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            # Construct enhanced prompt with player context
            enhanced_prompt = (
                f"Context: We are discussing {player_name}, a soccer striker. "
                f"Player Statistics: {metrics_context}. "
                f"User Question: {prompt}"
            )
            
            # Generate response
            response = model.generate_content(enhanced_prompt)
            return response.text


        def main():
            st.title("⚽ Soccer Performance Analysis")
            
            # Load striker data
            striker_data = pd.read_csv('striker.csv')
            
            # Ensure only numeric columns are used for z-scores
            numeric_columns = striker_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'player_name']
            
            # Player selection altready defined
            selected_player = jogadores
            #selected_player = st.selectbox(
            #    "Select a Striker", 
            #    options=striker_data['player_name'].tolist()
            #)
                        
            # Get player's data
            player_info = striker_data[striker_data['player_name'] == selected_player].iloc[0]
            
            # Display player's metrics
            #st.subheader(f"Metrics for {selected_player}")
            #metrics_df = player_info[numeric_columns].to_frame('Z-Score').T
            #metrics_df.index = ['Performance Metrics']
            #st.dataframe(metrics_df)
            
            # Generate and display player description
            if selected_player: #st.button('Generate Player Description'):
                with st.spinner('Generating player description...'):

                    # Generate initial description
                    description = generate_player_description(player_info)
                    
                    # Initialize chat history with the description
                    initialize_chat_history(selected_player, description)
                    
                    # Display initial description
                    st.info(description)
            
            # Chat interface (only appears after description is generated)
            if 'messages' in st.session_state:
                st.subheader(f"Can I help you with something else?")
                
                # Chat input (moved up as requested)
                if prompt := st.chat_input("Ask a question about the player"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        response = generate_chat_response(prompt, selected_player, player_info)
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            main()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

    elif posição == ("FULL BACK"):
        #Plotar Primeiro Gráfico - Radar de Percentis do Jogador na liga:
        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Percentiles)</h3>", unsafe_allow_html=True)
        Lateral_Charts = pd.read_csv('variable_df_adj_final_per.csv')
        Lateral_Charts_1 = Lateral_Charts[(Lateral_Charts['player_name']==jogadores)&(Lateral_Charts['role_2']==posição)]
        columns_to_rename = {
            col: col.replace('_per', '') for col in Lateral_Charts.columns if '_per' in col
        }
        # Renaming the columns in the DataFrame
        Lateral_Charts_1.rename(columns=columns_to_rename, inplace=True)
        #Collecting data to plot
        metrics = Lateral_Charts_1.iloc[:, np.r_[27, 15, 26, 34, 19, 31, 28, 33, 32]].reset_index(drop=True)
        metrics_list = metrics.iloc[0].tolist()
        #Collecting clube
        clube = Lateral_Charts_1.iat[0, 4]
        
        ## parameter names
        params = metrics.columns.tolist()

        ## range values
        ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

        ## parameter value
        values = metrics_list

        ## title values
        title = dict(
            title_name=jogadores,
            title_color = 'blue',
            subtitle_name= (posição),
            subtitle_color='#344D94',
            title_name_2=clube,
            title_color_2 = 'blue',
            subtitle_name_2='2023',
            subtitle_color_2='#344D94',
            title_fontsize=20,
            subtitle_fontsize=18,
        )            

        ## endnote 
        endnote = ""

        ## instantiate object
        radar = Radar()

        ## instantiate object -- changing fontsize
        radar=Radar(fontfamily='Cursive', range_fontsize=13)
        radar=Radar(fontfamily='Cursive', label_fontsize=15)

        ## plot radar -- filename and dpi
        fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=[('#B6282F', 0.65), ('#344D94', 0.65)], 
                                title=title, endnote=endnote, dpi=600)
        st.pyplot(fig)

        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        #Plotar Segundo Gráfico - Dispersão dos jogadores da mesma posição na liga em eixo único:

        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)


        # Dynamically create the HTML string with the 'jogadores' variable
        title_html = f"<h3 style='text-align: center; font-weight: bold; color: blue;'>{jogadores}</h3>"

        # Use the dynamically created HTML string in st.markdown
        st.markdown(title_html, unsafe_allow_html=True)

        #st.markdown("<h3 style='text-align: center;'>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)
        # Collecting data
        Lateral_Charts_2 = pd.read_csv('variable_df_adj_final_z.csv')
        Lateral_Charts_2 = Lateral_Charts_2[(Lateral_Charts['role_2']==posição)]

        #Collecting data to plot
        metrics = Lateral_Charts_2.iloc[:, np.r_[26, 15, 25, 33, 18, 30, 27, 32, 31]].reset_index(drop=True)
        metrics_involvement = metrics.iloc[:, 0].tolist()
        metrics_active_defence = metrics.iloc[:, 1].tolist()
        metrics_intelligent_defence = metrics.iloc[:, 2].tolist()
        metrics_territorial_dominance = metrics.iloc[:, 3].tolist()
        metrics_chance_prevention = metrics.iloc[:, 4].tolist()
        metrics_progression = metrics.iloc[:, 5].tolist()
        metrics_passing_quality = metrics.iloc[:, 6].tolist()
        metrics_run_quality = metrics.iloc[:, 7].tolist()
        metrics_providing_teammates = metrics.iloc[:, 8].tolist()
        metrics_y = [0] * len(metrics_involvement)

        # The specific data point you want to highlight
        highlight = Lateral_Charts_2[(Lateral_Charts_2['player_name']==jogadores)]
        highlight = highlight.iloc[:, np.r_[26, 15, 25, 33, 18, 30, 27, 32, 31]].reset_index(drop=True)
        highlight_involvement = highlight.iloc[:, 0].tolist()
        highlight_active_defence = highlight.iloc[:, 1].tolist()
        highlight_intelligent_defence = highlight.iloc[:, 2].tolist()
        highlight_territorial_dominance = highlight.iloc[:, 3].tolist()
        highlight_chance_prevention = highlight.iloc[:, 4].tolist()
        highlight_progression = highlight.iloc[:, 5].tolist()
        highlight_passing_quality = highlight.iloc[:, 6].tolist()
        highlight_run_quality = highlight.iloc[:, 7].tolist()
        highlight_providing_teammates = highlight.iloc[:, 8].tolist()
        highlight_y = 0

        # Computing the selected player specific values
        highlight_involvement_value = pd.DataFrame(highlight_involvement).reset_index(drop=True)
        highlight_active_defence_value = pd.DataFrame(highlight_active_defence).reset_index(drop=True)
        highlight_intelligent_defence_value = pd.DataFrame(highlight_intelligent_defence).reset_index(drop=True)
        highlight_territorial_dominance_value = pd.DataFrame(highlight_territorial_dominance).reset_index(drop=True)
        highlight_chance_prevention_value = pd.DataFrame(highlight_chance_prevention).reset_index(drop=True)
        highlight_progression_value = pd.DataFrame(highlight_progression).reset_index(drop=True)
        highlight_passing_quality_value = pd.DataFrame(highlight_passing_quality).reset_index(drop=True)
        highlight_run_quality_value = pd.DataFrame(highlight_run_quality).reset_index(drop=True)
        highlight_providing_teammates_value = pd.DataFrame(highlight_providing_teammates).reset_index(drop=True)

        highlight_involvement_value = highlight_involvement_value.iat[0,0]
        highlight_active_defence_value = highlight_active_defence_value.iat[0,0]
        highlight_intelligent_defence_value = highlight_intelligent_defence_value.iat[0,0]
        highlight_territorial_dominance_value = highlight_territorial_dominance_value.iat[0,0]
        highlight_chance_prevention_value = highlight_chance_prevention_value.iat[0,0]
        highlight_progression_value = highlight_progression_value.iat[0,0]
        highlight_passing_quality_value = highlight_passing_quality_value.iat[0,0]
        highlight_run_quality_value = highlight_run_quality_value.iat[0,0]
        highlight_providing_teammates_value = highlight_providing_teammates_value.iat[0,0]

        # Computing the min and max value across all lists using a generator expression
        min_value = min(min(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_territorial_dominance, metrics_chance_prevention, metrics_progression,
                                            metrics_passing_quality, metrics_run_quality, metrics_providing_teammates])
        min_value = min_value - 0.1
        max_value = max(max(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_territorial_dominance, metrics_chance_prevention, metrics_progression,
                                            metrics_passing_quality, metrics_run_quality, metrics_providing_teammates])
        max_value = max_value + 0.1

        # Create two subplots vertically aligned with separate x-axes
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1)
        ax.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        #Collecting Additional Information
        # Load the saved DataFrame from "Lateral_ranking.csv"
        lateral_ranking_df = pd.read_csv("variable_df_adj_final_rank.csv")

        # Building the Extended Title"
        rows_count = lateral_ranking_df[lateral_ranking_df['role_2'] == "FULL BACK"].shape[0]
        involvement_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'involvement'].values
        involvement_ranking_value = involvement_ranking_value[0].astype(int)
        output_str = f"({involvement_ranking_value}/{rows_count})"
        full_title_involvement = f"Involvement {output_str} {highlight_involvement_value}"

        # Building the Extended Title"
        active_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'active_defence'].values
        active_defence_ranking_value = active_defence_ranking_value[0].astype(int)
        output_str = f"({active_defence_ranking_value}/{rows_count})"
        full_title_active_defence = f"Active defence {output_str} {highlight_active_defence_value}"

        # Building the Extended Title"
        intelligent_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'intelligent_defence'].values
        intelligent_defence_ranking_value = intelligent_defence_ranking_value[0].astype(int)
        output_str = f"({intelligent_defence_ranking_value}/{rows_count})"
        full_title_intelligent_defence = f"Intelligent defence {output_str} {highlight_intelligent_defence_value}"

        # Building the Extended Title"
        territorial_dominance_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'territorial_dominance'].values
        territorial_dominance_ranking_value = territorial_dominance_ranking_value[0].astype(int)
        output_str = f"({territorial_dominance_ranking_value}/{rows_count})"
        full_title_territorial_dominance = f"Territorial dominance {output_str} {highlight_intelligent_defence_value}"
        
        # Building the Extended Title"
        chance_prevention_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'chance_prevention'].values
        chance_prevention_ranking_value = chance_prevention_ranking_value[0].astype(int)
        output_str = f"({chance_prevention_ranking_value}/{rows_count})"
        full_title_chance_prevention = f"Chance prevention {output_str} {highlight_chance_prevention_value}"

        # Building the Extended Title"
        progression_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'progression'].values
        progression_ranking_value = progression_ranking_value[0].astype(int)
        output_str = f"({progression_ranking_value}/{rows_count})"
        full_title_progression = f"Progression {output_str} {highlight_progression_value}"

        # Building the Extended Title"
        passing_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'passing_quality'].values
        passing_quality_ranking_value = passing_quality_ranking_value[0].astype(int)
        output_str = f"({passing_quality_ranking_value}/{rows_count})"
        full_title_passing_quality = f"Passing quality {output_str} {highlight_passing_quality_value}"

        # Building the Extended Title"
        run_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'run_quality'].values
        run_quality_ranking_value = run_quality_ranking_value[0].astype(int)
        output_str = f"({run_quality_ranking_value}/{rows_count})"
        full_title_run_quality = f"Run quality {output_str} {highlight_run_quality_value}"

        # Building the Extended Title"
        providing_teammates_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "FULL BACK"), 'providing_teammates'].values
        providing_teammates_ranking_value = providing_teammates_ranking_value[0].astype(int)
        output_str = f"({providing_teammates_ranking_value}/{rows_count})"
        full_title_providing_teammates = f"Providing teammates {output_str} {highlight_providing_teammates_value}"

        # Plot the first scatter plot in the first subplot
        ax1.scatter(metrics_involvement, metrics_y, color='deepskyblue')
        ax1.scatter(highlight_involvement, highlight_y, color='blue', s=60)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title(full_title_involvement, fontsize=9, fontweight='bold')
        ax1.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax2.scatter(metrics_active_defence, metrics_y, color='deepskyblue')
        ax2.scatter(highlight_active_defence, highlight_y, color='blue', s=60)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(full_title_active_defence, fontsize=9, fontweight='bold')
        ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax3.scatter(metrics_intelligent_defence, metrics_y, color='deepskyblue')
        ax3.scatter(highlight_intelligent_defence, highlight_y, color='blue', s=60)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(full_title_intelligent_defence, fontsize=9, fontweight='bold')
        ax3.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax3.xaxis.set_major_locator(ticker.NullLocator())
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax4.scatter(metrics_territorial_dominance, metrics_y, color='deepskyblue')
        ax4.scatter(highlight_territorial_dominance, highlight_y, color='blue', s=60)            
        ax4.get_yaxis().set_visible(False)
        ax4.set_title(full_title_territorial_dominance, fontsize=9, fontweight='bold')
        ax4.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax4.xaxis.set_major_locator(ticker.NullLocator())
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax5.scatter(metrics_chance_prevention, metrics_y, color='deepskyblue')
        ax5.scatter(highlight_chance_prevention, highlight_y, color='blue', s=60)            
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(full_title_chance_prevention, fontsize=9, fontweight='bold')
        ax5.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax5.xaxis.set_major_locator(ticker.NullLocator())
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax6.scatter(metrics_progression, metrics_y, color='deepskyblue')
        ax6.scatter(highlight_progression, highlight_y, color='blue', s=60)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(full_title_progression, fontsize=9, fontweight='bold')
        ax6.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax6.xaxis.set_major_locator(ticker.NullLocator())
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax7.scatter(metrics_passing_quality, metrics_y, color='deepskyblue')
        ax7.scatter(highlight_passing_quality, highlight_y, color='blue', s=60)
        ax7.get_yaxis().set_visible(False)
        ax7.set_title(full_title_passing_quality, fontsize=9, fontweight='bold')
        ax7.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax7.xaxis.set_major_locator(ticker.NullLocator())
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax7.spines['left'].set_visible(False)
        ax7.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax8.scatter(metrics_run_quality, metrics_y, color='deepskyblue')
        ax8.scatter(highlight_run_quality, highlight_y, color='blue', s=60)
        ax8.get_yaxis().set_visible(False)
        ax8.set_title(full_title_run_quality, fontsize=9, fontweight='bold')
        ax8.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax8.xaxis.set_major_locator(ticker.NullLocator())
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['left'].set_visible(False)
        ax8.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()

        # Plot the second scatter plot in the second subplot
        ax9.scatter(metrics_providing_teammates, metrics_y, color='deepskyblue', label='Other players in the league')
        ax9.scatter(highlight_providing_teammates, highlight_y, color='blue', s=60, label=jogadores)            
        ax9.get_yaxis().set_visible(False)
        ax9.set_title(full_title_providing_teammates, fontsize=9, fontweight='bold')
        ax9.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax9.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.spines['bottom'].set_visible(False)
        ax9.spines['left'].set_visible(False)
        ax9.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        ax9.legend(loc='right', bbox_to_anchor=(0.2, -6), fontsize="6", frameon=False)
        plt.show()

        st.pyplot(fig)

##########################################################################################################################
##########################################################################################################################
#####################################################################################################################            
#####################################################################################################################            

        def describe(thresholds, words, value):
            """
            Converts a z-score to a descriptive word based on predefined thresholds.
            
            Args:
                thresholds (list): Ordered list of z-score thresholds
                words (list): Corresponding descriptive words
                value (float): Z-score to categorize
            
            Returns:
                str: Descriptive word for the z-score
            """
            # Ensure value is converted to float and handle potential errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                return "unavailable"
            
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return words[i]
            return words[-1]

        def describe_level(value):
            """
            Describes a player's metric performance level.
            
            Args:
                value (float): Z-score of a metric
            
            Returns:
                str: Descriptive performance level
            """
            thresholds = [1.5, 1, 0.5, -0.5, -1]
            words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
            return describe(thresholds, words, value)

        def generate_player_description(player_data):
            """
            Generate a descriptive summary of a player's performance.
            
            Args:
                player_data (pd.Series): Row containing player's z-scores
            
            Returns:
                str: Descriptive narrative about the player
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare descriptive metrics (only for numeric columns)
            metrics_description = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            prompt = (
                f"Please use the statistical description to give a concise, 4 sentence summary of {player_data['player_name']}'s playing style, strengths and weaknesses. "
                f"Statistical Context: {metrics_description}. "
                "The first sentence should use varied language to give an overview of the player. "
                "The second sentence should describe the player's specific strengths based on the metrics. "
                "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
                "Finally, summarise exactly how the player compares to others in the same position. "
            )
            
            response = model.generate_content(prompt)
            return response.text

        def initialize_chat_history(player_name, initial_description):
            """
            Initialize the chat history with the player description.
            
            Args:
                player_name (str): Name of the selected player
                initial_description (str): Initial AI-generated player description
            """
            # Reset chat history
            st.session_state.messages = [
                {"role": "assistant", "content": f"Player Analysis for {player_name}:"},
                {"role": "assistant", "content": initial_description}
            ]

        def generate_chat_response(prompt, player_name, player_data):
            """
            Generate a response from Gemini based on the chat prompt.
            
            Args:
                prompt (str): User's chat input
                player_name (str): Name of the selected player
                player_data (pd.Series): Player's statistical data
            
            Returns:
                str: AI-generated response
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare additional context from player metrics
            metrics_context = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            # Construct enhanced prompt with player context
            enhanced_prompt = (
                f"Context: We are discussing {player_name}, a full back. "
                f"Player Statistics: {metrics_context}. "
                f"User Question: {prompt}"
            )
            
            # Generate response
            response = model.generate_content(enhanced_prompt)
            return response.text


        def main():
            st.title("⚽ Soccer Performance Analysis")
            
            # Load striker data
            full_back_data = pd.read_csv('lateral.csv')
            
            # Ensure only numeric columns are used for z-scores
            numeric_columns = full_back_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'player_name']
            
            # Player selection altready defined
            selected_player = jogadores
            #selected_player = st.selectbox(
            #    "Select a Striker", 
            #    options=striker_data['player_name'].tolist()
            #)
                        
            # Get player's data
            player_info = full_back_data[full_back_data['player_name'] == selected_player].iloc[0]
            
            # Display player's metrics
            #st.subheader(f"Metrics for {selected_player}")
            #metrics_df = player_info[numeric_columns].to_frame('Z-Score').T
            #metrics_df.index = ['Performance Metrics']
            #st.dataframe(metrics_df)
            
            # Generate and display player description
            if selected_player: #st.button('Generate Player Description'):
                with st.spinner('Generating player description...'):

                    # Generate initial description
                    description = generate_player_description(player_info)
                    
                    # Initialize chat history with the description
                    initialize_chat_history(selected_player, description)
                    
                    # Display initial description
                    st.info(description)
            
            # Chat interface (only appears after description is generated)
            if 'messages' in st.session_state:
                st.subheader(f"Can I help you with something else?")
                
                # Chat input (moved up as requested)
                if prompt := st.chat_input("Ask a question about the player"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        response = generate_chat_response(prompt, selected_player, player_info)
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            main()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

    if posição == ("DEFENCE"):
        #Plotar Primeiro Gráfico - Radar de Percentis do Jogador na liga:
        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Percentiles)</h3>", unsafe_allow_html=True)
        Lateral_Charts = pd.read_csv('variable_df_adj_final_per.csv')
        Lateral_Charts_1 = Lateral_Charts[(Lateral_Charts['player_name']==jogadores)&(Lateral_Charts['role_2']==posição)]
        columns_to_rename = {
            col: col.replace('_per', '') for col in Lateral_Charts.columns if '_per' in col
        }
        # Renaming the columns in the DataFrame
        Lateral_Charts_1.rename(columns=columns_to_rename, inplace=True)
        #Collecting data to plot
        metrics = Lateral_Charts_1.iloc[:, np.r_[27, 15, 26, 34, 19, 21, 17, 20, 31]].reset_index(drop=True)
        metrics_list = metrics.iloc[0].tolist()
        #Collecting clube
        clube = Lateral_Charts_1.iat[0, 4]
        
        ## parameter names
        params = metrics.columns.tolist()

        ## range values
        ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

        ## parameter value
        values = metrics_list

        ## title values
        title = dict(
            title_name=jogadores,
            title_color = 'blue',
            subtitle_name= (posição),
            subtitle_color='#344D94',
            title_name_2=clube,
            title_color_2 = 'blue',
            subtitle_name_2='2023',
            subtitle_color_2='#344D94',
            title_fontsize=20,
            subtitle_fontsize=18,
        )            

        ## endnote 
        endnote = ""

        ## instantiate object
        radar = Radar()

        ## instantiate object -- changing fontsize
        radar=Radar(fontfamily='Cursive', range_fontsize=13)
        radar=Radar(fontfamily='Cursive', label_fontsize=15)

        ## plot radar -- filename and dpi
        fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=[('#B6282F', 0.65), ('#344D94', 0.65)], 
                                title=title, endnote=endnote, dpi=600)
        st.pyplot(fig)

        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        #Plotar Segundo Gráfico - Dispersão dos jogadores da mesma posição na liga em eixo único:

        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)


        # Dynamically create the HTML string with the 'jogadores' variable
        title_html = f"<h3 style='text-align: center; font-weight: bold; color: blue;'>{jogadores}</h3>"

        # Use the dynamically created HTML string in st.markdown
        st.markdown(title_html, unsafe_allow_html=True)

        #st.markdown("<h3 style='text-align: center;'>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)
        # Collecting data
        Lateral_Charts_2 = pd.read_csv('variable_df_adj_final_z.csv')
        Lateral_Charts_2 = Lateral_Charts_2[(Lateral_Charts['role_2']==posição)]

        #Collecting data to plot
        metrics = Lateral_Charts_2.iloc[:, np.r_[26, 15, 25, 33, 18, 20, 16, 19, 30]].reset_index(drop=True)
        metrics_involvement = metrics.iloc[:, 0].tolist()
        metrics_active_defence = metrics.iloc[:, 1].tolist()
        metrics_intelligent_defence = metrics.iloc[:, 2].tolist()
        metrics_territorial_dominance = metrics.iloc[:, 3].tolist()
        metrics_chance_prevention = metrics.iloc[:, 4].tolist()
        metrics_defensive_heading = metrics.iloc[:, 5].tolist()
        metrics_aerial_threat = metrics.iloc[:, 6].tolist()
        metrics_composure = metrics.iloc[:, 7].tolist()
        metrics_progression = metrics.iloc[:, 8].tolist()
        metrics_y = [0] * len(metrics_involvement)

        # The specific data point you want to highlight
        highlight = Lateral_Charts_2[(Lateral_Charts_2['player_name']==jogadores)]
        highlight = highlight.iloc[:, np.r_[26, 15, 25, 33, 18, 20, 16, 19, 30]].reset_index(drop=True)
        highlight_involvement = highlight.iloc[:, 0].tolist()
        highlight_active_defence = highlight.iloc[:, 1].tolist()
        highlight_intelligent_defence = highlight.iloc[:, 2].tolist()
        highlight_territorial_dominance = highlight.iloc[:, 3].tolist()
        highlight_chance_prevention = highlight.iloc[:, 4].tolist()
        highlight_defensive_heading = highlight.iloc[:, 5].tolist()
        highlight_aerial_threat = highlight.iloc[:, 6].tolist()
        highlight_composure = highlight.iloc[:, 7].tolist()
        highlight_progression = highlight.iloc[:, 8].tolist()
        highlight_y = 0

        # Computing the selected player specific values
        highlight_involvement_value = pd.DataFrame(highlight_involvement).reset_index(drop=True)
        highlight_active_defence_value = pd.DataFrame(highlight_active_defence).reset_index(drop=True)
        highlight_intelligent_defence_value = pd.DataFrame(highlight_intelligent_defence).reset_index(drop=True)
        highlight_territorial_dominance_value = pd.DataFrame(highlight_territorial_dominance).reset_index(drop=True)
        highlight_chance_prevention_value = pd.DataFrame(highlight_chance_prevention).reset_index(drop=True)
        highlight_defensive_heading_value = pd.DataFrame(highlight_defensive_heading).reset_index(drop=True)
        highlight_aerial_threat_value = pd.DataFrame(highlight_aerial_threat).reset_index(drop=True)
        highlight_composure_value = pd.DataFrame(highlight_composure).reset_index(drop=True)
        highlight_progression_value = pd.DataFrame(highlight_progression).reset_index(drop=True)

        highlight_involvement_value = highlight_involvement_value.iat[0,0]
        highlight_active_defence_value = highlight_active_defence_value.iat[0,0]
        highlight_intelligent_defence_value = highlight_intelligent_defence_value.iat[0,0]
        highlight_territorial_dominance_value = highlight_territorial_dominance_value.iat[0,0]
        highlight_chance_prevention_value = highlight_chance_prevention_value.iat[0,0]
        highlight_defensive_heading_value = highlight_defensive_heading_value.iat[0,0]
        highlight_aerial_threat_value = highlight_aerial_threat_value.iat[0,0]
        highlight_composure_value = highlight_composure_value.iat[0,0]
        highlight_progression_value = highlight_progression_value.iat[0,0]

        # Computing the min and max value across all lists using a generator expression
        min_value = min(min(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_territorial_dominance, metrics_chance_prevention,metrics_defensive_heading, 
                                            metrics_aerial_threat, metrics_composure, metrics_progression])
        min_value = min_value - 0.1
        max_value = max(max(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_territorial_dominance, metrics_chance_prevention,metrics_defensive_heading,
                                            metrics_aerial_threat, metrics_composure, metrics_progression])
        max_value = max_value + 0.1

        # Create two subplots vertically aligned with separate x-axes
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1)
        ax.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        #Collecting Additional Information
        # Load the saved DataFrame from "Lateral_ranking.csv"
        lateral_ranking_df = pd.read_csv("variable_df_adj_final_rank.csv")

        # Building the Extended Title"
        rows_count = lateral_ranking_df[lateral_ranking_df['role_2'] == "DEFENCE"].shape[0]
        involvement_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'involvement'].values
        involvement_ranking_value = involvement_ranking_value[0].astype(int)
        output_str = f"({involvement_ranking_value}/{rows_count})"
        full_title_involvement = f"Involvement {output_str} {highlight_involvement_value}"

        # Building the Extended Title"
        active_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'active_defence'].values
        active_defence_ranking_value = active_defence_ranking_value[0].astype(int)
        output_str = f"({active_defence_ranking_value}/{rows_count})"
        full_title_active_defence = f"Active defence {output_str} {highlight_active_defence_value}"

        # Building the Extended Title"
        intelligent_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'intelligent_defence'].values
        intelligent_defence_ranking_value = intelligent_defence_ranking_value[0].astype(int)
        output_str = f"({intelligent_defence_ranking_value}/{rows_count})"
        full_title_intelligent_defence = f"Intelligent defence {output_str} {highlight_intelligent_defence_value}"

        # Building the Extended Title"
        territorial_dominance_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'territorial_dominance'].values
        territorial_dominance_ranking_value = territorial_dominance_ranking_value[0].astype(int)
        output_str = f"({territorial_dominance_ranking_value}/{rows_count})"
        full_title_territorial_dominance = f"Territorial dominance {output_str} {highlight_territorial_dominance_value}"
        
        # Building the Extended Title"
        chance_prevention_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'chance_prevention'].values
        chance_prevention_ranking_value = chance_prevention_ranking_value[0].astype(int)
        output_str = f"({chance_prevention_ranking_value}/{rows_count})"
        full_title_chance_prevention = f"Chance prevention {output_str} {highlight_chance_prevention_value}"

        # Building the Extended Title"
        defensive_heading_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'defensive_heading'].values
        defensive_heading_ranking_value = defensive_heading_ranking_value[0].astype(int)
        output_str = f"({defensive_heading_ranking_value}/{rows_count})"
        full_title_defensive_heading = f"Defensive heading {output_str} {highlight_defensive_heading_value}"

        # Building the Extended Title"
        aerial_threat_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'aerial_threat'].values
        aerial_threat_ranking_value = aerial_threat_ranking_value[0].astype(int)
        output_str = f"({aerial_threat_ranking_value}/{rows_count})"
        full_title_aerial_threat = f"Aerial threat {output_str} {highlight_aerial_threat_value}"

        # Building the Extended Title"
        composure_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'composure'].values
        composure_ranking_value = composure_ranking_value[0].astype(int)
        output_str = f"({composure_ranking_value}/{rows_count})"
        full_title_composure = f"Composure {output_str} {highlight_composure_value}"

        # Building the Extended Title"
        progression_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "DEFENCE"), 'progression'].values
        progression_ranking_value = progression_ranking_value[0].astype(int)
        output_str = f"({progression_ranking_value}/{rows_count})"
        full_title_progression = f"Progression {output_str} {highlight_progression_value}"

        # Plot the first scatter plot in the first subplot
        ax1.scatter(metrics_involvement, metrics_y, color='deepskyblue')
        ax1.scatter(highlight_involvement, highlight_y, color='blue', s=60)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title(full_title_involvement, fontsize=9, fontweight='bold')
        ax1.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax2.scatter(metrics_active_defence, metrics_y, color='deepskyblue')
        ax2.scatter(highlight_active_defence, highlight_y, color='blue', s=60)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(full_title_active_defence, fontsize=9, fontweight='bold')
        ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax3.scatter(metrics_intelligent_defence, metrics_y, color='deepskyblue')
        ax3.scatter(highlight_intelligent_defence, highlight_y, color='blue', s=60)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(full_title_intelligent_defence, fontsize=9, fontweight='bold')
        ax3.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax3.xaxis.set_major_locator(ticker.NullLocator())
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax4.scatter(metrics_territorial_dominance, metrics_y, color='deepskyblue')
        ax4.scatter(highlight_territorial_dominance, highlight_y, color='blue', s=60)            
        ax4.get_yaxis().set_visible(False)
        ax4.set_title(full_title_territorial_dominance, fontsize=9, fontweight='bold')
        ax4.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax4.xaxis.set_major_locator(ticker.NullLocator())
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax5.scatter(metrics_chance_prevention, metrics_y, color='deepskyblue')
        ax5.scatter(highlight_chance_prevention, highlight_y, color='blue', s=60)            
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(full_title_chance_prevention, fontsize=9, fontweight='bold')
        ax5.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax5.xaxis.set_major_locator(ticker.NullLocator())
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot


        # Plot the second scatter plot in the second subplot
        ax6.scatter(metrics_defensive_heading, metrics_y, color='deepskyblue')
        ax6.scatter(highlight_defensive_heading, highlight_y, color='blue', s=60)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(full_title_defensive_heading, fontsize=9, fontweight='bold')
        ax6.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax6.xaxis.set_major_locator(ticker.NullLocator())
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax7.scatter(metrics_aerial_threat, metrics_y, color='deepskyblue')
        ax7.scatter(highlight_aerial_threat, highlight_y, color='blue', s=60)
        ax7.get_yaxis().set_visible(False)
        ax7.set_title(full_title_aerial_threat, fontsize=9, fontweight='bold')
        ax7.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax7.xaxis.set_major_locator(ticker.NullLocator())
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax7.spines['left'].set_visible(False)
        ax7.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax8.scatter(metrics_composure, metrics_y, color='deepskyblue')
        ax8.scatter(highlight_composure, highlight_y, color='blue', s=60)            
        ax8.get_yaxis().set_visible(False)
        ax8.set_title(full_title_composure, fontsize=9, fontweight='bold')
        ax8.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax8.xaxis.set_major_locator(ticker.NullLocator())
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['left'].set_visible(False)
        ax8.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax9.scatter(metrics_progression, metrics_y, color='deepskyblue', label='Other players in the league')
        ax9.scatter(highlight_progression, highlight_y, color='blue', s=60, label=jogadores)
        ax9.get_yaxis().set_visible(False)
        ax9.set_title(full_title_progression, fontsize=9, fontweight='bold')
        ax9.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax9.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.spines['bottom'].set_visible(False)
        ax9.spines['left'].set_visible(False)
        ax9.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        ax9.legend(loc='right', bbox_to_anchor=(0.2, -4.32), fontsize="6", frameon=False)
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()

        st.pyplot(fig)

#####################################################################################################################            
#####################################################################################################################            

        def describe(thresholds, words, value):
            """
            Converts a z-score to a descriptive word based on predefined thresholds.
            
            Args:
                thresholds (list): Ordered list of z-score thresholds
                words (list): Corresponding descriptive words
                value (float): Z-score to categorize
            
            Returns:
                str: Descriptive word for the z-score
            """
            # Ensure value is converted to float and handle potential errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                return "unavailable"
            
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return words[i]
            return words[-1]

        def describe_level(value):
            """
            Describes a player's metric performance level.
            
            Args:
                value (float): Z-score of a metric
            
            Returns:
                str: Descriptive performance level
            """
            thresholds = [1.5, 1, 0.5, -0.5, -1]
            words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
            return describe(thresholds, words, value)

        def generate_player_description(player_data):
            """
            Generate a descriptive summary of a player's performance.
            
            Args:
                player_data (pd.Series): Row containing player's z-scores
            
            Returns:
                str: Descriptive narrative about the player
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare descriptive metrics (only for numeric columns)
            metrics_description = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            prompt = (
                f"Please use the statistical description to give a concise, 4 sentence summary of {player_data['player_name']}'s playing style, strengths and weaknesses. "
                f"Statistical Context: {metrics_description}. "
                "The first sentence should use varied language to give an overview of the player. "
                "The second sentence should describe the player's specific strengths based on the metrics. "
                "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
                "Finally, summarise exactly how the player compares to others in the same position. "
            )
            
            response = model.generate_content(prompt)
            return response.text

        def initialize_chat_history(player_name, initial_description):
            """
            Initialize the chat history with the player description.
            
            Args:
                player_name (str): Name of the selected player
                initial_description (str): Initial AI-generated player description
            """
            # Reset chat history
            st.session_state.messages = [
                {"role": "assistant", "content": f"Player Analysis for {player_name}:"},
                {"role": "assistant", "content": initial_description}
            ]

        def generate_chat_response(prompt, player_name, player_data):
            """
            Generate a response from Gemini based on the chat prompt.
            
            Args:
                prompt (str): User's chat input
                player_name (str): Name of the selected player
                player_data (pd.Series): Player's statistical data
            
            Returns:
                str: AI-generated response
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare additional context from player metrics
            metrics_context = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            # Construct enhanced prompt with player context
            enhanced_prompt = (
                f"Context: We are discussing {player_name}, a defence player. "
                f"Player Statistics: {metrics_context}. "
                f"User Question: {prompt}"
            )
            
            # Generate response
            response = model.generate_content(enhanced_prompt)
            return response.text


        def main():
            st.title("⚽ Soccer Performance Analysis")
            
            # Load striker data
            defence_data = pd.read_csv('defence.csv')
            
            # Ensure only numeric columns are used for z-scores
            numeric_columns = defence_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'player_name']
            
            # Player selection altready defined
            selected_player = jogadores
            #selected_player = st.selectbox(
            #    "Select a Striker", 
            #    options=striker_data['player_name'].tolist()
            #)
                        
            # Get player's data
            player_info = defence_data[defence_data['player_name'] == selected_player].iloc[0]
            
            # Display player's metrics
            #st.subheader(f"Metrics for {selected_player}")
            #metrics_df = player_info[numeric_columns].to_frame('Z-Score').T
            #metrics_df.index = ['Performance Metrics']
            #st.dataframe(metrics_df)
            
            # Generate and display player description
            if selected_player: #st.button('Generate Player Description'):
                with st.spinner('Generating player description...'):

                    # Generate initial description
                    description = generate_player_description(player_info)
                    
                    # Initialize chat history with the description
                    initialize_chat_history(selected_player, description)
                    
                    # Display initial description
                    st.info(description)
            
            # Chat interface (only appears after description is generated)
            if 'messages' in st.session_state:
                st.subheader(f"Can I help you with something else?")
                
                # Chat input (moved up as requested)
                if prompt := st.chat_input("Ask a question about the player"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        response = generate_chat_response(prompt, selected_player, player_info)
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            main()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

    if posição == ("MIDFIELDER"):
        #Plotar Primeiro Gráfico - Radar de Percentis do Jogador na liga:
        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Percentiles)</h3>", unsafe_allow_html=True)
        Lateral_Charts = pd.read_csv('variable_df_adj_final_per.csv')
        Lateral_Charts_1 = Lateral_Charts[(Lateral_Charts['player_name']==jogadores)&(Lateral_Charts['role_2']==posição)]
        columns_to_rename = {
            col: col.replace('_per', '') for col in Lateral_Charts.columns if '_per' in col
        }
        # Renaming the columns in the DataFrame
        Lateral_Charts_1.rename(columns=columns_to_rename, inplace=True)
        #Collecting data to plot
        metrics = Lateral_Charts_1.iloc[:, np.r_[27, 16, 26, 31, 28, 23, 32, 18]].reset_index(drop=True)
        metrics_list = metrics.iloc[0].tolist()
        #Collecting clube
        clube = Lateral_Charts_1.iat[0, 4]
        
        ## parameter names
        params = metrics.columns.tolist()

        ## range values
        ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

        ## parameter value
        values = metrics_list

        ## title values
        title = dict(
            title_name=jogadores,
            title_color = 'blue',
            subtitle_name= (posição),
            subtitle_color='#344D94',
            title_name_2=clube,
            title_color_2 = 'blue',
            subtitle_name_2='2023',
            subtitle_color_2='#344D94',
            title_fontsize=20,
            subtitle_fontsize=18,
        )            

        ## endnote 
        endnote = ""

        ## instantiate object
        radar = Radar()

        ## instantiate object -- changing fontsize
        radar=Radar(fontfamily='Cursive', range_fontsize=13)
        radar=Radar(fontfamily='Cursive', label_fontsize=15)

        ## plot radar -- filename and dpi
        fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=[('#B6282F', 0.65), ('#344D94', 0.65)], 
                                title=title, endnote=endnote, dpi=600)
        st.pyplot(fig)

        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        #Plotar Segundo Gráfico - Dispersão dos jogadores da mesma posição na liga em eixo único:

        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)


        # Dynamically create the HTML string with the 'jogadores' variable
        title_html = f"<h3 style='text-align: center; font-weight: bold; color: blue;'>{jogadores}</h3>"

        # Use the dynamically created HTML string in st.markdown
        st.markdown(title_html, unsafe_allow_html=True)

        #st.markdown("<h3 style='text-align: center;'>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)
        # Collecting data
        Lateral_Charts_2 = pd.read_csv('variable_df_adj_final_z.csv')
        Lateral_Charts_2 = Lateral_Charts_2[(Lateral_Charts['role_2']==posição)]

        #Collecting data to plot
        metrics = Lateral_Charts_2.iloc[:, np.r_[26, 15, 25, 30, 27, 22, 31, 17]].reset_index(drop=True)
        metrics_involvement = metrics.iloc[:, 0].tolist()
        metrics_active_defence = metrics.iloc[:, 1].tolist()
        metrics_intelligent_defence = metrics.iloc[:, 2].tolist()
        metrics_progression = metrics.iloc[:, 3].tolist()
        metrics_passing_quality = metrics.iloc[:, 4].tolist()
        metrics_effectiveness = metrics.iloc[:, 5].tolist()
        metrics_providing_teammates = metrics.iloc[:, 6].tolist()
        metrics_box_threat = metrics.iloc[:, 7].tolist()
        metrics_y = [0] * len(metrics_involvement)

        # The specific data point you want to highlight
        highlight = Lateral_Charts_2[(Lateral_Charts_2['player_name']==jogadores)]
        highlight = highlight.iloc[:, np.r_[26, 15, 25, 30, 27, 22, 31, 17]].reset_index(drop=True)
        highlight_involvement = highlight.iloc[:, 0].tolist()
        highlight_active_defence = highlight.iloc[:, 1].tolist()
        highlight_intelligent_defence = highlight.iloc[:, 2].tolist()
        highlight_progression = highlight.iloc[:, 3].tolist()
        highlight_passing_quality = highlight.iloc[:, 4].tolist()
        highlight_effectiveness = highlight.iloc[:, 5].tolist()
        highlight_providing_teammates = highlight.iloc[:, 6].tolist()
        highlight_box_threat = highlight.iloc[:, 7].tolist()
        highlight_y = 0

        # Computing the selected player specific values
        highlight_involvement_value = pd.DataFrame(highlight_involvement).reset_index(drop=True)
        highlight_active_defence_value = pd.DataFrame(highlight_active_defence).reset_index(drop=True)
        highlight_intelligent_defence_value = pd.DataFrame(highlight_intelligent_defence).reset_index(drop=True)
        highlight_progression_value = pd.DataFrame(highlight_progression).reset_index(drop=True)
        highlight_passing_quality_value = pd.DataFrame(highlight_passing_quality).reset_index(drop=True)
        highlight_effectiveness_value = pd.DataFrame(highlight_effectiveness).reset_index(drop=True)
        highlight_providing_teammates_value = pd.DataFrame(highlight_providing_teammates).reset_index(drop=True)
        highlight_box_threat_value = pd.DataFrame(highlight_box_threat).reset_index(drop=True)

        highlight_involvement_value = highlight_involvement_value.iat[0,0]
        highlight_active_defence_value = highlight_active_defence_value.iat[0,0]
        highlight_intelligent_defence_value = highlight_intelligent_defence_value.iat[0,0]
        highlight_progression_value = highlight_progression_value.iat[0,0]
        highlight_passing_quality_value = highlight_passing_quality_value.iat[0,0]
        highlight_effectiveness_value = highlight_effectiveness_value.iat[0,0]
        highlight_providing_teammates_value = highlight_providing_teammates_value.iat[0,0]
        highlight_box_threat_value = highlight_box_threat_value.iat[0,0]

        # Computing the min and max value across all lists using a generator expression
        min_value = min(min(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_progression, metrics_passing_quality, 
                                            metrics_effectiveness, metrics_providing_teammates, metrics_box_threat])
        min_value = min_value - 0.1
        max_value = max(max(lst) for lst in [metrics_involvement, metrics_active_defence, metrics_intelligent_defence, 
                                            metrics_progression, metrics_passing_quality,
                                            metrics_effectiveness, metrics_providing_teammates, metrics_box_threat])
        max_value = max_value + 0.1

        # Create two subplots vertically aligned with separate x-axes
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1)
        ax.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        #Collecting Additional Information
        # Load the saved DataFrame from "Lateral_ranking.csv"
        lateral_ranking_df = pd.read_csv("variable_df_adj_final_rank.csv")

        # Building the Extended Title"
        rows_count = lateral_ranking_df[lateral_ranking_df['role_2'] == "MIDFIELDER"].shape[0]
        involvement_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'involvement'].values
        involvement_ranking_value = involvement_ranking_value[0].astype(int)
        output_str = f"({involvement_ranking_value}/{rows_count})"
        full_title_involvement = f"Involvement {output_str} {highlight_involvement_value}"

        # Building the Extended Title"
        active_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'active_defence'].values
        active_defence_ranking_value = active_defence_ranking_value[0].astype(int)
        output_str = f"({active_defence_ranking_value}/{rows_count})"
        full_title_active_defence = f"Active defence {output_str} {highlight_active_defence_value}"

        # Building the Extended Title"
        intelligent_defence_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'intelligent_defence'].values
        intelligent_defence_ranking_value = intelligent_defence_ranking_value[0].astype(int)
        output_str = f"({intelligent_defence_ranking_value}/{rows_count})"
        full_title_intelligent_defence = f"Intelligent defence {output_str} {highlight_intelligent_defence_value}"

        # Building the Extended Title"
        progression_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'progression'].values
        progression_ranking_value = progression_ranking_value[0].astype(int)
        output_str = f"({progression_ranking_value}/{rows_count})"
        full_title_progression = f"Progression {output_str} {highlight_progression_value}"
        
        # Building the Extended Title"
        passing_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'passing_quality'].values
        passing_quality_ranking_value = passing_quality_ranking_value[0].astype(int)
        output_str = f"({passing_quality_ranking_value}/{rows_count})"
        full_title_passing_quality = f"Passing quality {output_str} {highlight_passing_quality_value}"

        # Building the Extended Title"
        effectiveness_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'effectiveness'].values
        effectiveness_ranking_value = effectiveness_ranking_value[0].astype(int)
        output_str = f"({effectiveness_ranking_value}/{rows_count})"
        full_title_effectiveness = f"Effectiveness {output_str} {highlight_effectiveness_value}"

        # Building the Extended Title"
        providing_teammates_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'providing_teammates'].values
        providing_teammates_ranking_value = providing_teammates_ranking_value[0].astype(int)
        output_str = f"({providing_teammates_ranking_value}/{rows_count})"
        full_title_providing_teammates = f"Providing teammates {output_str} {highlight_providing_teammates_value}"

        # Building the Extended Title"
        box_threat_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "MIDFIELDER"), 'box_threat'].values
        box_threat_ranking_value = box_threat_ranking_value[0].astype(int)
        output_str = f"({box_threat_ranking_value}/{rows_count})"
        full_title_box_threat = f"Box threat {output_str} {highlight_box_threat_value}"

        # Plot the first scatter plot in the first subplot
        ax1.scatter(metrics_involvement, metrics_y, color='deepskyblue')
        ax1.scatter(highlight_involvement, highlight_y, color='blue', s=60)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title(full_title_involvement, fontsize=9, fontweight='bold')
        ax1.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax2.scatter(metrics_active_defence, metrics_y, color='deepskyblue')
        ax2.scatter(highlight_active_defence, highlight_y, color='blue', s=60)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(full_title_active_defence, fontsize=9, fontweight='bold')
        ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax3.scatter(metrics_intelligent_defence, metrics_y, color='deepskyblue')
        ax3.scatter(highlight_intelligent_defence, highlight_y, color='blue', s=60)            
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(full_title_intelligent_defence, fontsize=9, fontweight='bold')
        ax3.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax3.xaxis.set_major_locator(ticker.NullLocator())
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax4.scatter(metrics_progression, metrics_y, color='deepskyblue')
        ax4.scatter(highlight_progression, highlight_y, color='blue', s=60)            
        ax4.get_yaxis().set_visible(False)
        ax4.set_title(full_title_progression, fontsize=9, fontweight='bold')
        ax4.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax4.xaxis.set_major_locator(ticker.NullLocator())
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax5.scatter(metrics_passing_quality, metrics_y, color='deepskyblue')
        ax5.scatter(highlight_passing_quality, highlight_y, color='blue', s=60)
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(full_title_passing_quality, fontsize=9, fontweight='bold')
        ax5.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax5.xaxis.set_major_locator(ticker.NullLocator())
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax6.scatter(metrics_effectiveness, metrics_y, color='deepskyblue')
        ax6.scatter(highlight_effectiveness, highlight_y, color='blue', s=60)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(full_title_effectiveness, fontsize=9, fontweight='bold')
        ax6.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax6.xaxis.set_major_locator(ticker.NullLocator())
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax7.scatter(metrics_providing_teammates, metrics_y, color='deepskyblue', label='Other players in the league')
        ax7.scatter(highlight_providing_teammates, highlight_y, color='blue', s=60, label=jogadores)
        ax7.get_yaxis().set_visible(False)
        ax7.set_title(full_title_providing_teammates, fontsize=9, fontweight='bold')
        ax7.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax7.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax7.spines['left'].set_visible(False)
        ax7.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax8.scatter(metrics_box_threat, metrics_y, color='deepskyblue', label='Other players in the league')
        ax8.scatter(highlight_box_threat, highlight_y, color='blue', s=60, label=jogadores)
        ax8.get_yaxis().set_visible(False)
        ax8.set_title(full_title_box_threat, fontsize=9, fontweight='bold')
        ax8.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax8.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['left'].set_visible(False)
        ax8.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        ax8.legend(loc='right', bbox_to_anchor=(0.2, -2.5), fontsize="6", frameon=False)
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()

        st.pyplot(fig)
        
#####################################################################################################################            
#####################################################################################################################            

        def describe(thresholds, words, value):
            """
            Converts a z-score to a descriptive word based on predefined thresholds.
            
            Args:
                thresholds (list): Ordered list of z-score thresholds
                words (list): Corresponding descriptive words
                value (float): Z-score to categorize
            
            Returns:
                str: Descriptive word for the z-score
            """
            # Ensure value is converted to float and handle potential errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                return "unavailable"
            
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return words[i]
            return words[-1]

        def describe_level(value):
            """
            Describes a player's metric performance level.
            
            Args:
                value (float): Z-score of a metric
            
            Returns:
                str: Descriptive performance level
            """
            thresholds = [1.5, 1, 0.5, -0.5, -1]
            words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
            return describe(thresholds, words, value)

        def generate_player_description(player_data):
            """
            Generate a descriptive summary of a player's performance.
            
            Args:
                player_data (pd.Series): Row containing player's z-scores
            
            Returns:
                str: Descriptive narrative about the player
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare descriptive metrics (only for numeric columns)
            metrics_description = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            prompt = (
                f"Please use the statistical description to give a concise, 4 sentence summary of {player_data['player_name']}'s playing style, strengths and weaknesses. "
                f"Statistical Context: {metrics_description}. "
                "The first sentence should use varied language to give an overview of the player. "
                "The second sentence should describe the player's specific strengths based on the metrics. "
                "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
                "Finally, summarise exactly how the player compares to others in the same position. "
            )
            
            response = model.generate_content(prompt)
            return response.text

        def initialize_chat_history(player_name, initial_description):
            """
            Initialize the chat history with the player description.
            
            Args:
                player_name (str): Name of the selected player
                initial_description (str): Initial AI-generated player description
            """
            # Reset chat history
            st.session_state.messages = [
                {"role": "assistant", "content": f"Player Analysis for {player_name}:"},
                {"role": "assistant", "content": initial_description}
            ]

        def generate_chat_response(prompt, player_name, player_data):
            """
            Generate a response from Gemini based on the chat prompt.
            
            Args:
                prompt (str): User's chat input
                player_name (str): Name of the selected player
                player_data (pd.Series): Player's statistical data
            
            Returns:
                str: AI-generated response
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare additional context from player metrics
            metrics_context = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            # Construct enhanced prompt with player context
            enhanced_prompt = (
                f"Context: We are discussing {player_name}, a midfielder. "
                f"Player Statistics: {metrics_context}. "
                f"User Question: {prompt}"
            )
            
            # Generate response
            response = model.generate_content(enhanced_prompt)
            return response.text


        def main():
            st.title("⚽ Soccer Performance Analysis")
            
            # Load striker data
            midfielder_data = pd.read_csv('midfielder.csv')
            
            # Ensure only numeric columns are used for z-scores
            numeric_columns = midfielder_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'player_name']
            
            # Player selection altready defined
            selected_player = jogadores
            #selected_player = st.selectbox(
            #    "Select a Striker", 
            #    options=striker_data['player_name'].tolist()
            #)
                        
            # Get player's data
            player_info = midfielder_data[midfielder_data['player_name'] == selected_player].iloc[0]
            
            # Display player's metrics
            #st.subheader(f"Metrics for {selected_player}")
            #metrics_df = player_info[numeric_columns].to_frame('Z-Score').T
            #metrics_df.index = ['Performance Metrics']
            #st.dataframe(metrics_df)
            
            # Generate and display player description
            if selected_player: #st.button('Generate Player Description'):
                with st.spinner('Generating player description...'):

                    # Generate initial description
                    description = generate_player_description(player_info)
                    
                    # Initialize chat history with the description
                    initialize_chat_history(selected_player, description)
                    
                    # Display initial description
                    st.info(description)
            
            # Chat interface (only appears after description is generated)
            if 'messages' in st.session_state:
                st.subheader(f"Can I help you with something else?")
                
                # Chat input (moved up as requested)
                if prompt := st.chat_input("Ask a question about the player"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        response = generate_chat_response(prompt, selected_player, player_info)
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            main()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

    if posição == ("WINGER"):
        #Plotar Primeiro Gráfico - Radar de Percentis do Jogador na liga:
        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Percentiles)</h3>", unsafe_allow_html=True)
        Lateral_Charts = pd.read_csv('variable_df_adj_final_per.csv')
        Lateral_Charts_1 = Lateral_Charts[(Lateral_Charts['player_name']==jogadores)&(Lateral_Charts['role_2']==posição)]
        columns_to_rename = {
            col: col.replace('_per', '') for col in Lateral_Charts.columns if '_per' in col
        }
        # Renaming the columns in the DataFrame
        Lateral_Charts_1.rename(columns=columns_to_rename, inplace=True)
        #Collecting data to plot
        metrics = Lateral_Charts_1.iloc[:, np.r_[27, 30, 28, 33, 22, 23, 32, 18, 24]].reset_index(drop=True)
        metrics_list = metrics.iloc[0].tolist()
        #Collecting clube
        clube = Lateral_Charts_1.iat[0, 4]
        
        ## parameter names
        params = metrics.columns.tolist()

        ## range values
        ranges = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]

        ## parameter value
        values = metrics_list

        ## title values
        title = dict(
            title_name=jogadores,
            title_color = 'blue',
            subtitle_name= (posição),
            subtitle_color='#344D94',
            title_name_2=clube,
            title_color_2 = 'blue',
            subtitle_name_2='2023',
            subtitle_color_2='#344D94',
            title_fontsize=20,
            subtitle_fontsize=18,
        )            

        ## endnote 
        endnote = ""

        ## instantiate object
        radar = Radar()

        ## instantiate object -- changing fontsize
        radar=Radar(fontfamily='Cursive', range_fontsize=13)
        radar=Radar(fontfamily='Cursive', label_fontsize=15)

        ## plot radar -- filename and dpi
        fig, ax = radar.plot_radar(ranges=ranges, params=params, values=values, radar_color=[('#B6282F', 0.65), ('#344D94', 0.65)], 
                                title=title, endnote=endnote, dpi=600)
        st.pyplot(fig)

        #################################################################################################################################
        #################################################################################################################################
        #################################################################################################################################
        
        #Plotar Segundo Gráfico - Dispersão dos jogadores da mesma posição na liga em eixo único:

        st.markdown("<h3 style='text-align: center; color: blue; '>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)


        # Dynamically create the HTML string with the 'jogadores' variable
        title_html = f"<h3 style='text-align: center; font-weight: bold; color: blue;'>{jogadores}</h3>"

        # Use the dynamically created HTML string in st.markdown
        st.markdown(title_html, unsafe_allow_html=True)

        #st.markdown("<h3 style='text-align: center;'>Players' Attributes in Brasileirão 2023 (in Z-scores)</h3>", unsafe_allow_html=True)
        # Collecting data
        Lateral_Charts_2 = pd.read_csv('variable_df_adj_final_z.csv')
        Lateral_Charts_2 = Lateral_Charts_2[(Lateral_Charts['role_2']==posição)]

        #Collecting data to plot
        metrics = Lateral_Charts_2.iloc[:, np.r_[26, 29, 27, 32, 21, 22, 31, 17, 23]].reset_index(drop=True)
        metrics_involvement = metrics.iloc[:, 0].tolist()
        metrics_pressing = metrics.iloc[:, 1].tolist()
        metrics_passing_quality = metrics.iloc[:, 2].tolist()
        metrics_run_quality = metrics.iloc[:, 3].tolist()
        metrics_dribbling = metrics.iloc[:, 4].tolist()
        metrics_effectiveness = metrics.iloc[:, 5].tolist()
        metrics_providing_teammates = metrics.iloc[:, 6].tolist()
        metrics_box_threat = metrics.iloc[:, 7].tolist()
        metrics_finishing = metrics.iloc[:, 8].tolist()
        metrics_y = [0] * len(metrics_involvement)

        # The specific data point you want to highlight
        highlight = Lateral_Charts_2[(Lateral_Charts_2['player_name']==jogadores)]
        highlight = highlight.iloc[:, np.r_[26, 29, 27, 32, 21, 22, 31, 17, 23]].reset_index(drop=True)
        highlight_involvement = highlight.iloc[:, 0].tolist()
        highlight_pressing = highlight.iloc[:, 1].tolist()
        highlight_passing_quality = highlight.iloc[:, 2].tolist()
        highlight_run_quality = highlight.iloc[:, 3].tolist()
        highlight_dribbling = highlight.iloc[:, 4].tolist()
        highlight_effectiveness = highlight.iloc[:, 5].tolist()
        highlight_providing_teammates = highlight.iloc[:, 6].tolist()
        highlight_box_threat = highlight.iloc[:, 7].tolist()
        highlight_finishing = highlight.iloc[:, 8].tolist()
        highlight_y = 0

        # Computing the selected player specific values
        highlight_involvement_value = pd.DataFrame(highlight_involvement).reset_index(drop=True)
        highlight_pressing_value = pd.DataFrame(highlight_pressing).reset_index(drop=True)
        highlight_passing_quality_value = pd.DataFrame(highlight_passing_quality).reset_index(drop=True)
        highlight_run_quality_value = pd.DataFrame(highlight_run_quality).reset_index(drop=True)
        highlight_dribbling_value = pd.DataFrame(highlight_dribbling).reset_index(drop=True)
        highlight_effectiveness_value = pd.DataFrame(highlight_effectiveness).reset_index(drop=True)
        highlight_providing_teammates_value = pd.DataFrame(highlight_providing_teammates).reset_index(drop=True)
        highlight_box_threat_value = pd.DataFrame(highlight_box_threat).reset_index(drop=True)
        highlight_finishing_value = pd.DataFrame(highlight_finishing).reset_index(drop=True)

        highlight_involvement_value = highlight_involvement_value.iat[0,0]
        highlight_pressing_value = highlight_pressing_value.iat[0,0]
        highlight_passing_quality_value = highlight_passing_quality_value.iat[0,0]
        highlight_run_quality_value = highlight_run_quality_value.iat[0,0]
        highlight_dribbling_value = highlight_dribbling_value.iat[0,0]
        highlight_effectiveness_value = highlight_effectiveness_value.iat[0,0]
        highlight_providing_teammates_value = highlight_providing_teammates_value.iat[0,0]
        highlight_box_threat_value = highlight_box_threat_value.iat[0,0]
        highlight_finishing_value = highlight_finishing_value.iat[0,0]

        # Computing the min and max value across all lists using a generator expression
        min_value = min(min(lst) for lst in [metrics_involvement, metrics_pressing, metrics_passing_quality, 
                                            metrics_run_quality, metrics_dribbling,metrics_effectiveness, 
                                            metrics_providing_teammates, metrics_box_threat, metrics_finishing])
        min_value = min_value - 0.1
        max_value = max(max(lst) for lst in [metrics_involvement, metrics_pressing, metrics_passing_quality, 
                                            metrics_run_quality, metrics_dribbling,metrics_effectiveness,
                                            metrics_providing_teammates, metrics_box_threat, metrics_finishing])
        max_value = max_value + 0.1

        # Create two subplots vertically aligned with separate x-axes
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1)
        ax.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        #Collecting Additional Information
        # Load the saved DataFrame from "Lateral_ranking.csv"
        lateral_ranking_df = pd.read_csv("variable_df_adj_final_rank.csv")

        # Building the Extended Title"
        rows_count = lateral_ranking_df[lateral_ranking_df['role_2'] == "WINGER"].shape[0]
        involvement_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'involvement'].values
        involvement_ranking_value = involvement_ranking_value[0].astype(int)
        output_str = f"({involvement_ranking_value}/{rows_count})"
        full_title_involvement = f"Involvement {output_str} {highlight_involvement_value}"

        # Building the Extended Title"
        pressing_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'pressing'].values
        pressing_ranking_value = pressing_ranking_value[0].astype(int)
        output_str = f"({pressing_ranking_value}/{rows_count})"
        full_title_pressing = f"Pressing {output_str} {highlight_pressing_value}"

        # Building the Extended Title"
        passing_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'passing_quality'].values
        passing_quality_ranking_value = passing_quality_ranking_value[0].astype(int)
        output_str = f"({passing_quality_ranking_value}/{rows_count})"
        full_title_passing_quality = f"Passing quality {output_str} {highlight_passing_quality_value}"

        # Building the Extended Title"
        run_quality_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'run_quality'].values
        run_quality_ranking_value = run_quality_ranking_value[0].astype(int)
        output_str = f"({run_quality_ranking_value}/{rows_count})"
        full_title_run_quality = f"Run quality {output_str} {highlight_run_quality_value}"
        
        # Building the Extended Title"
        dribbling_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'dribbling'].values
        dribbling_ranking_value = dribbling_ranking_value[0].astype(int)
        output_str = f"({dribbling_ranking_value}/{rows_count})"
        full_title_dribbling = f"Dribbling {output_str} {highlight_dribbling_value}"

        # Building the Extended Title"
        effectiveness_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'effectiveness'].values
        effectiveness_ranking_value = effectiveness_ranking_value[0].astype(int)
        output_str = f"({effectiveness_ranking_value}/{rows_count})"
        full_title_effectiveness = f"Effectiveness {output_str} {highlight_effectiveness_value}"

        # Building the Extended Title"
        providing_teammates_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'providing_teammates'].values
        providing_teammates_ranking_value = providing_teammates_ranking_value[0].astype(int)
        output_str = f"({providing_teammates_ranking_value}/{rows_count})"
        full_title_providing_teammates = f"Providing teammates {output_str} {highlight_providing_teammates_value}"

        # Building the Extended Title"
        box_threat_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'box_threat'].values
        box_threat_ranking_value = box_threat_ranking_value[0].astype(int)
        output_str = f"({box_threat_ranking_value}/{rows_count})"
        full_title_box_threat = f"Box threat {output_str} {highlight_box_threat_value}"

        # Building the Extended Title"
        finishing_ranking_value = lateral_ranking_df.loc[(lateral_ranking_df['player_name'] == jogadores) & 
                                                            (lateral_ranking_df['role_2'] == "WINGER"), 'finishing'].values
        finishing_ranking_value = finishing_ranking_value[0].astype(int)
        output_str = f"({finishing_ranking_value}/{rows_count})"
        full_title_finishing = f"Finishing {output_str} {highlight_finishing_value}"

        # Plot the first scatter plot in the first subplot
        ax1.scatter(metrics_involvement, metrics_y, color='deepskyblue')
        ax1.scatter(highlight_involvement, highlight_y, color='blue', s=60)
        ax1.get_yaxis().set_visible(False)
        ax1.set_title(full_title_involvement, fontsize=9, fontweight='bold')
        ax1.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax2.scatter(metrics_pressing, metrics_y, color='deepskyblue')
        ax2.scatter(highlight_pressing, highlight_y, color='blue', s=60)
        ax2.get_yaxis().set_visible(False)
        ax2.set_title(full_title_pressing, fontsize=9, fontweight='bold')
        ax2.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax2.xaxis.set_major_locator(ticker.NullLocator())
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax3.scatter(metrics_passing_quality, metrics_y, color='deepskyblue')
        ax3.scatter(highlight_passing_quality, highlight_y, color='blue', s=60)            
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(full_title_passing_quality, fontsize=9, fontweight='bold')
        ax3.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax3.xaxis.set_major_locator(ticker.NullLocator())
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax4.scatter(metrics_run_quality, metrics_y, color='deepskyblue')
        ax4.scatter(highlight_run_quality, highlight_y, color='blue', s=60)            
        ax4.get_yaxis().set_visible(False)
        ax4.set_title(full_title_run_quality, fontsize=9, fontweight='bold')
        ax4.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax4.xaxis.set_major_locator(ticker.NullLocator())
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax5.scatter(metrics_dribbling, metrics_y, color='deepskyblue')
        ax5.scatter(highlight_dribbling, highlight_y, color='blue', s=60)
        ax5.get_yaxis().set_visible(False)
        ax5.set_title(full_title_dribbling, fontsize=9, fontweight='bold')
        ax5.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax5.xaxis.set_major_locator(ticker.NullLocator())
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax6.scatter(metrics_effectiveness, metrics_y, color='deepskyblue')
        ax6.scatter(highlight_effectiveness, highlight_y, color='blue', s=60)
        ax6.get_yaxis().set_visible(False)
        ax6.set_title(full_title_effectiveness, fontsize=9, fontweight='bold')
        ax6.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax6.xaxis.set_major_locator(ticker.NullLocator())
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax7.scatter(metrics_providing_teammates, metrics_y, color='deepskyblue')
        ax7.scatter(highlight_providing_teammates, highlight_y, color='blue', s=60)
        ax7.get_yaxis().set_visible(False)
        ax7.set_title(full_title_providing_teammates, fontsize=9, fontweight='bold')
        ax7.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax7.xaxis.set_major_locator(ticker.NullLocator())
        ax7.spines['top'].set_visible(False)
        ax7.spines['right'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax7.spines['left'].set_visible(False)
        ax7.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax8.scatter(metrics_box_threat, metrics_y, color='deepskyblue', label='Other players in the league')
        ax8.scatter(highlight_box_threat, highlight_y, color='blue', s=60, label=jogadores)
        ax8.get_yaxis().set_visible(False)
        ax8.set_title(full_title_box_threat, fontsize=9, fontweight='bold')
        ax8.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax8.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        ax8.spines['bottom'].set_visible(False)
        ax8.spines['left'].set_visible(False)
        ax8.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot

        # Plot the second scatter plot in the second subplot
        ax9.scatter(metrics_finishing, metrics_y, color='deepskyblue', label='Other players in the league')
        ax9.scatter(highlight_finishing, highlight_y, color='blue', s=60, label=jogadores)
        ax9.get_yaxis().set_visible(False)
        ax9.set_title(full_title_finishing, fontsize=9, fontweight='bold')
        ax9.axhline(y=0, color='grey', linewidth=1, alpha=0.4)
        ax9.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax9.spines['top'].set_visible(False)
        ax9.spines['right'].set_visible(False)
        ax9.spines['bottom'].set_visible(False)
        ax9.spines['left'].set_visible(False)
        ax9.set_xlim(min_value, max_value)  # Set the same x-axis scale for each plot
        ax9.legend(loc='right', bbox_to_anchor=(0.2, -2.5), fontsize="6", frameon=False)
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.show()

        st.pyplot(fig)

##########################################################################################################################
##########################################################################################################################

        def describe(thresholds, words, value):
            """
            Converts a z-score to a descriptive word based on predefined thresholds.
            
            Args:
                thresholds (list): Ordered list of z-score thresholds
                words (list): Corresponding descriptive words
                value (float): Z-score to categorize
            
            Returns:
                str: Descriptive word for the z-score
            """
            # Ensure value is converted to float and handle potential errors
            try:
                value = float(value)
            except (ValueError, TypeError):
                return "unavailable"
            
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    return words[i]
            return words[-1]

        def describe_level(value):
            """
            Describes a player's metric performance level.
            
            Args:
                value (float): Z-score of a metric
            
            Returns:
                str: Descriptive performance level
            """
            thresholds = [1.5, 1, 0.5, -0.5, -1]
            words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
            return describe(thresholds, words, value)

        def generate_player_description(player_data):
            """
            Generate a descriptive summary of a player's performance.
            
            Args:
                player_data (pd.Series): Row containing player's z-scores
            
            Returns:
                str: Descriptive narrative about the player
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare descriptive metrics (only for numeric columns)
            metrics_description = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            prompt = (
                f"Please use the statistical description to give a concise, 4 sentence summary of {player_data['player_name']}'s playing style, strengths and weaknesses. "
                f"Statistical Context: {metrics_description}. "
                "The first sentence should use varied language to give an overview of the player. "
                "The second sentence should describe the player's specific strengths based on the metrics. "
                "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
                "Finally, summarise exactly how the player compares to others in the same position. "
            )
            
            response = model.generate_content(prompt)
            return response.text

        def initialize_chat_history(player_name, initial_description):
            """
            Initialize the chat history with the player description.
            
            Args:
                player_name (str): Name of the selected player
                role_2 (str): Postion of the player
                initial_description (str): Initial AI-generated player description
            """
            # Reset chat history
            st.session_state.messages = [
                {"role": "assistant", "content": f"Player Analysis for {player_name}:"},
                {"role": "assistant", "content": initial_description}
            ]

        def generate_chat_response(prompt, player_name, player_data):
            """
            Generate a response from Gemini based on the chat prompt.
            
            Args:
                prompt (str): User's chat input
                player_name (str): Name of the selected player
                player_data (pd.Series): Player's statistical data
            
            Returns:
                str: AI-generated response
            """
            # Configure Gemini API
            #genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
            #model = genai.GenerativeModel('gemini-pro')
            # Try to get API key from Streamlit secrets, with a fallback
            api_key = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY'))
            if not api_key:
                st.error("No API key found. Please configure the Gemini API key.")
            else:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
            
            # Prepare additional context from player metrics
            metrics_context = " ".join([
                f"{metric.replace('_', ' ').title()}: {describe_level(value)} ({value:.2f} z-score)"
                for metric, value in player_data.items() 
                if metric not in ['player_name'] and pd.api.types.is_numeric_dtype(type(value))
            ])
            
            # Construct enhanced prompt with player context
            enhanced_prompt = (
                f"Context: We are discussing {player_name}, a winger. "
                f"Player Statistics: {metrics_context}. "
                f"User Question: {prompt}"
            )
            
            # Generate response
            response = model.generate_content(enhanced_prompt)
            return response.text


        def main():
            st.title("⚽ Soccer Performance Analysis")
            
            # Load striker data
            winger_data = pd.read_csv('winger.csv')
            
            # Ensure only numeric columns are used for z-scores
            numeric_columns = winger_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col != 'player_name']
            
            # Player selection altready defined
            selected_player = jogadores
            #selected_player = st.selectbox(
            #    "Select a Striker", 
            #    options=striker_data['player_name'].tolist()
            #)
                        
            # Get player's data
            player_info = winger_data[winger_data['player_name'] == selected_player].iloc[0]
            
            # Display player's metrics
            #st.subheader(f"Metrics for {selected_player}")
            #metrics_df = player_info[numeric_columns].to_frame('Z-Score').T
            #metrics_df.index = ['Performance Metrics']
            #st.dataframe(metrics_df)
            
            # Generate and display player description
            if selected_player: #st.button('Generate Player Description'):
                with st.spinner('Generating player description...'):

                    # Generate initial description
                    description = generate_player_description(player_info)
                    
                    # Initialize chat history with the description
                    initialize_chat_history(selected_player, description)
                    
                    # Display initial description
                    st.info(description)
            
            # Chat interface (only appears after description is generated)
            if 'messages' in st.session_state:
                st.subheader(f"Can I help you with something else?")
                
                # Chat input (moved up as requested)
                if prompt := st.chat_input("Ask a question about the player"):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        response = generate_chat_response(prompt, selected_player, player_info)
                        st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            main()
