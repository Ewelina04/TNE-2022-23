
# cd C:\Users\User\Downloads

# python -m streamlit run C:\Users\User\Downloads\TimeAn2.py


vac_red = r"C:\Users\User\Downloads\PolarIs1_VaccRed.xlsx"

vac_tw = r"C:\Users\User\Downloads\Polaris2_2.xlsx"

cch_red = r"C:\Users\User\Downloads\Polaris3_2.xlsx"

cch_tw = r"C:\Users\User\Downloads\Polaris4_2.xlsx"

# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-talk")

from scipy.stats import pearsonr, pointbiserialr, spearmanr

import spacy
nlp = spacy.load('en_core_web_sm')

pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS


# functions

def clean_text(df, text_column):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df["content"] = new_texts
  return df

def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(collocations=False, max_words=100, colormap=colormap, width = width, height = height,
                background_color ='black',
                min_font_size = 16, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig, wordcloud.words_.keys()


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['clean_sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['clean_sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['clean_sentence_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2


import random

def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'crest'):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'Reds' #gist_heat
    dataframe['precis'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['precis'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['precis'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['precis'] >= int(lexeme_threshold)) & (dataframe['general #'] > 2) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with precis threshold equal to {lexeme_threshold}.')
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)
  st.write(f"There are {n_words} words.")
  figure_cloud, _ = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud


def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


@st.cache(allow_output_mutation=True)
def load_data(file_path, indx = True, indx_col = 0):
  '''Parameters:
  file_path: path to your excel or csv file with data,

  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True

  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
  return data


def lemmatization(dataframe, text_column):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].apply(str)):
    lemmas.append(" ".join([token.lemma_ for token in doc if (not token.is_punct and not token.is_stop and not token.like_num and len(token) > 1) ]))
  df[text_column +"_lemmatized"] = lemmas
  return df


def clean_text(df, text_column):
  import re
  new_texts = []
  for text in df[text_column]:
    text_list = str(text).lower().split(" ")
    new_string_list = []
    for word in text_list:
      if 'http' in word:
        word = "url"
      elif ('@' in word) and (len(word) > 1):
        word = "@"
      if (len(word) > 1) and not (word.isnumeric()):
        new_string_list.append(word)
    new_string = " ".join(new_string_list)
    new_string = re.sub("\d+", " ", new_string)
    new_string = new_string.replace('\n', ' ')
    new_string = new_string.replace('  ', ' ')
    new_string = new_string.strip()
    new_texts.append(new_string)
  df["content"] = new_texts
  return df


def get_valence_scores(data, lemmatized_column, affective_database, db_words = "Word"):
  '''Parameters:
  dataframe: dataframe with your data,

  lemmatized_column: str - name of a column in dataframe where word-lemmas are listed,

  affective_database_path: str - path to a file with affective database,

  db_words: str - name of a column in affective database where words are listed,
  '''
  dataframe = data.copy()
  #affective_database = load_data(affective_database_path)

  emotion_values = ["Valence_standardized"]
  used_cols = [db_words] + emotion_values

  affective_database_valence = affective_database[used_cols]
  affective_database_valence.set_index(db_words, inplace=True)
  affective_database_valence_words = affective_database[ (affective_database.Valence_standardized < -0.5) | (affective_database.Valence_standardized > 1) ][db_words].tolist()

  neg_valence_scores = []
  pos_valence_scores = []
  overall_scores = []

  for words_emo in dataframe[lemmatized_column]:
    #words_emo = row.split()
    words_emo = [w for w in words_emo.split() if w in affective_database_valence_words]
    if len(words_emo) > 0:
      scores = affective_database_valence.loc[words_emo]

      neg_score = scores.where(scores["Valence_standardized"].round(1) < -0.5).count()[0]
      neg_valence_scores.append(neg_score)

      pos_score = scores.where(scores["Valence_standardized"].round(1) > 1).count()[0]
      pos_valence_scores.append(pos_score)

      if pos_score > neg_score:
        overall_scores.append('positive')
      elif pos_score < neg_score:
        overall_scores.append('negative')
      elif pos_score == neg_score:
        overall_scores.append('neutral')

    else:
      neg_score=pos_score = np.NaN
      overall_scores.append('')
      neg_valence_scores.append(neg_score)
      pos_valence_scores.append(pos_score)

  dataframe["valence_score"] = overall_scores
  dataframe["valence_positive_count"] = pos_valence_scores
  dataframe["valence_negative_count"] = neg_valence_scores
  return dataframe




# app version
def user_stats_app(dataframe, source_column = 'source',
               ethos_column = 'ethos_label', pathos_column = 'pathos_label'):
  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  if not 'neutral' in dataframe[ethos_column]:
      dataframe[ethos_column] = dataframe[ethos_column].map(ethos_mapping)
  if not 'neutral' in dataframe[pathos_column]:
      dataframe[pathos_column] = dataframe[pathos_column].map(valence_mapping)

  df = pd.DataFrame(columns = ['user', 'text_n',
                               'ethos_n', 'ethos_support_n', 'ethos_attack_n',
                               'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
                             'ethos_percent', 'ethos_support_percent', 'ethos_attack_percent',
                             'pathos_percent', 'pathos_negative_percent', 'pathos_positive_percent'])
  users_list = []
  for i, u in enumerate(sources_list):
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    N_user = int(len(df_user))

    df_user_ethos = df_user.groupby(ethos_column, as_index = False)["sentence"].size()
    try:
      N_support = int(df_user_ethos[df_user_ethos[ethos_column] == 'support']['size'].iloc[0])
    except:
      N_support = 0
    try:
      N_attack = int(df_user_ethos[df_user_ethos[ethos_column] == 'attack']['size'].iloc[0])
    except:
      N_attack=0

    df_user_pathos = df_user.groupby(pathos_column, as_index = False)["sentence"].size()
    try:
      N_neg = int(df_user_pathos[df_user_pathos[pathos_column] == 'negative']['size'].iloc[0])
    except:
      N_neg = 0
    try:
      N_pos = int(df_user_pathos[df_user_pathos[pathos_column] == 'positive']['size'].iloc[0])
    except:
      N_pos = 0

    counts_list = [N_support+N_attack, N_support, N_attack, N_neg+N_pos, N_neg, N_pos]
    percent_list = list((np.array(counts_list) / N_user).round(3) * 100)
    df.loc[i] = [u] + [N_user] + counts_list + percent_list
  return df

def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


def user_rhetoric_v2(data, source_column = 'source', ethos_col = 'ethos_label',
                  pathos_col = 'pathos_label'):

  import warnings
  from pandas.core.common import SettingWithCopyWarning
  warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
  dataframe = data.copy()

  dataframe[source_column] = dataframe[source_column].apply(str)
  sources_list = dataframe[ ~(dataframe[source_column].isin(['nan', ''])) ][source_column].unique()
  metric_value = []
  users_list = []
  if not 'neutral' in dataframe[ethos_col]:
      dataframe[ethos_col] = dataframe[ethos_col].map(ethos_mapping)
  if not 'neutral' in dataframe[pathos_col]:
      dataframe[pathos_col] = dataframe[pathos_col].map(valence_mapping)

  map_ethos_weight = {'attack':-1, 'neutral':0, 'support':1}
  map_pathos_weight = {'negative':-1, 'neutral':0, 'positive':1}
  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    ethos_pathos_user = 0
    df_user_rhetoric = df_user.groupby([str(pathos_col), str(ethos_col)], as_index=False).size()
    # map weights
    df_user_rhetoric[pathos_col] = df_user_rhetoric[pathos_col].map(map_pathos_weight)
    df_user_rhetoric[ethos_col] = df_user_rhetoric[ethos_col].map(map_ethos_weight)

    ethos_pathos_sum_ids = []

    for id in df_user_rhetoric.index:
      ethos_pathos_val = np.sum(df_user_rhetoric.loc[id, str(pathos_col):str(ethos_col)].to_numpy())
      ethos_pathos_val = ethos_pathos_val * df_user_rhetoric.loc[id, 'size']
      ethos_pathos_sum_ids.append(ethos_pathos_val)

    ethos_pathos_user = np.sum(ethos_pathos_sum_ids)
    try:
        metric_value.append(int(ethos_pathos_user))
    except:
        metric_value.append(0)
  df = pd.DataFrame({'user': users_list, 'rhetoric_metric': metric_value})
  return df


def UsersExtreme(data):
    st.write("### Rhetoric Profiles")
    add_spacelines(2)
    df = data.copy()
    #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    radio_LEP_behavior = st.selectbox("Category of the analysed entities", ("speakers", "target entities"))
    add_spacelines(1)
    #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    radio_LEP_behavior_axis = st.radio("Choose the unit of y-axis", ("percentage", "number"))
    rhetoric_dims = ['ethos', 'pathos']

    if radio_LEP_behavior == "target entities":
        data_rh = user_rhetoric_v2(df, source_column = 'Target')
        user_stats_df = user_stats_app(df, source_column = 'Target')
    else:
        data_rh = user_rhetoric_v2(df)
        user_stats_df = user_stats_app(df)

    data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n','pathos_n', 'pathos_negative_n', 'pathos_positive_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)
    color = sns.color_palette("Reds", data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()+15)[::-1][:data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()] +sns.color_palette("Blues", 3)[2:] + sns.color_palette("Greens", data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()+20)[data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()*-1:] # + sns.color_palette("Greens", 15)[4:]

    if radio_LEP_behavior_axis == 'number':
        sns.set(font_scale=1.3, style='whitegrid')
        fig_rh_raw = sns.catplot(kind = 'count', data = data_rh, x = 'rhetoric_metric',
                    aspect = 1.8, palette = color, height = 6)
        for ax in fig_rh_raw.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
        if np.amax(data_rh.rhetoric_metric.value_counts().values) <= 50:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+6, 5), fontsize=16)
        elif np.amax(data_rh.rhetoric_metric.value_counts().values) <= 200:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 25), fontsize=16)
        elif np.amax(data_rh.rhetoric_metric.value_counts().values) < 400:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 50), fontsize=16)
        else:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+46, 100), fontsize=16)
        plt.ylabel('number of entities\n')
        plt.title("Rhetoric behaviour distribution\n")
        #plt.xticks(fontsize = 16)
        plt.xlabel('\nscore')
        plt.show()
        st.pyplot(fig_rh_raw)

    elif radio_LEP_behavior_axis == 'percentage':
        # change raw scores to percentages
        counts = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().values
        ids = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().index
        perc = (counts / len(data_rh)) * 100

        data_rh2 = pd.DataFrame({'rhetoric_metric': ids, 'percent':perc})
        data_rh2['percent'] = data_rh2['percent'].apply(lambda x: round(x, 1))
        sns.set(font_scale=1.3, style='whitegrid')
        fig_rh_percent = sns.catplot(kind = 'bar', data = data_rh2, x = 'rhetoric_metric',
                         y = 'percent',
                    aspect = 1.8, palette = color, height = 6, ci = None)
        for ax in fig_rh_percent.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
        if radio_LEP_behavior == "target entities":
            plt.yticks(np.arange(0, data_rh2.percent.max()+3, 2), fontsize = 16)
        else:
            plt.yticks(np.arange(0, data_rh2.percent.max()+6, 5), fontsize = 16)
        plt.ylabel('percentage of entities %\n', fontsize = 18)
        #plt.xticks(fontsize = 16)
        plt.title("Rhetoric behaviour distribution\n")
        plt.xlabel('\nscore')
        plt.show()
        st.pyplot(fig_rh_percent)

    add_spacelines(3)

    data_rh['standardized_scores'] = standardize(data_rh[['rhetoric_metric']])
    most_neg_users = data_rh.nsmallest(8, 'rhetoric_metric')
    most_pos_users = data_rh.nlargest(8, 'rhetoric_metric')

    most_neg_users_names = most_neg_users.user.tolist()
    most_pos_users_names = most_pos_users.user.tolist()

    users_rhet_cols = ['sentence', 'Target', 'pathos_label', 'ethos_label']
    with st.container():
        if radio_LEP_behavior == "target entities":
            head_neg_users = f'<p style="color:#D10000; font-size: 23px; font-weight: bold;">Most negative entities 😈</p>'
        else:
            head_neg_users = f'<p style="color:#D10000; font-size: 23px; font-weight: bold;">Most negative users 😈</p>'
        st.markdown(head_neg_users, unsafe_allow_html=True)
        col111, col222, col333, col444 = st.columns(4)
        with col111:
            st.write(f"**{most_neg_users_names[0]}**")
            col111.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[0], str(round(most_neg_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

        with col222:
            st.write(f"**{most_neg_users_names[1]}**")
            col222.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[1], str(round(most_neg_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

        with col333:
            st.write(f"**{most_neg_users_names[2]}**")
            col333.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[2], str(round(most_neg_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

        with col444:
            st.write(f"**{most_neg_users_names[3]}**")
            col444.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[3], str(round(most_neg_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

    add_spacelines(2)

    with st.container():
        col111, col222, col333, col444 = st.columns(4)
        with col111:
            st.write(f"**{most_neg_users_names[4]}**")
            col111.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[4], str(round(most_neg_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

        with col222:
            st.write(f"**{most_neg_users_names[5]}**")
            col222.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[5], str(round(most_neg_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

        with col333:
            st.write(f"**{most_neg_users_names[6]}**")
            col333.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[6], str(round(most_neg_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

        with col444:
            st.write(f"**{most_neg_users_names[7]}**")
            col444.metric('Rhetoric behaviour score', most_neg_users['rhetoric_metric'].iloc[7], str(round(most_neg_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

        add_spacelines(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if radio_LEP_behavior == "target entities":
            neg_users_to_df = st.radio("Choose name to see details about the entity \n", most_neg_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"sentences targeted at: **{neg_users_to_df}** ")
            st.dataframe(df[df.Target == str(neg_users_to_df)].set_index("source")[users_rhet_cols])
        else:
            neg_users_to_df = st.radio("Choose name to see details about the user \n", most_neg_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"sentences posted by: **{neg_users_to_df}** ")
            st.dataframe(df[df.source == str(neg_users_to_df)].set_index("source")[users_rhet_cols])
        add_spacelines(1)

    user_stats_df_user1 = user_stats_df[user_stats_df['user'] == str(neg_users_to_df)]
    with st.container():
        if radio_LEP_behavior == "target entities":
            st.write(f"##### Users' rhetoric strategy to speak about {neg_users_to_df}")
        else:
            st.write(f"##### {neg_users_to_df}'s rhetoric strategy")
        add_spacelines(1)
        col222, col333 = st.columns(2)
        with col222:
            st.write(f"**Ethos strategy**")
            col222.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_attack_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

        with col333:
            st.write(f"**Pathos strategy**")
            col333.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_negative_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

        strat_user_val_neg = [round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1),
                          round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1)]
        strat_user_val_neg_max = np.max(strat_user_val_neg)
        add_spacelines(1)
        strategy_lep_user_neg = []
        if radio_LEP_behavior == "target entities":
            if strat_user_val_neg[0] == strat_user_val_neg_max:
                strategy_lep_user_neg.append('ethos')
                #st.error(f"Users' negativity in speaking about **{neg_users_to_df}** comes mostly from **ethos**")
            if strat_user_val_neg[1] == strat_user_val_neg_max:
                strategy_lep_user_neg.append('pathos')
            strategy_lep_user_neg = " and ".join(strategy_lep_user_neg)
            st.error(f"Users' negativity in speaking about **{neg_users_to_df}** comes mostly from **{strategy_lep_user_neg}**")
        else:
            if strat_user_val_neg[0] == strat_user_val_neg_max:
                strategy_lep_user_neg.append('ethos')
                #st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **ethos**")
            if strat_user_val_neg[1] == strat_user_val_neg_max:
                strategy_lep_user_neg.append('pathos')
            strategy_lep_user_neg = " and ".join(strategy_lep_user_neg)
            st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **{strategy_lep_user_neg}**")

    add_spacelines(1)
    st.write(" **************************************************************************** ")
    add_spacelines(1)

    with st.container():
        if radio_LEP_behavior == "target entities":
            head_pos_users = f'<p style="color:#00A90D; font-size: 23px; font-weight: bold;">Most positive entities 😀</p>'
        else:
            head_pos_users = f'<p style="color:#00A90D; font-size: 23px; font-weight: bold;">Most positive users 😀</p>'
        st.markdown(head_pos_users, unsafe_allow_html=True)
        col11, col22, col33, col44 = st.columns(4)

        with col11:
            st.write(f"**{most_pos_users_names[0]}**")
            col11.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[0], str(round(most_pos_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

        with col22:
            st.write(f"**{most_pos_users_names[1]}**")
            col22.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[1], str(round(most_pos_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

        with col33:
            st.write(f"**{most_pos_users_names[2]}**")
            col33.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[2], str(round(most_pos_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

        with col44:
            st.write(f"**{most_pos_users_names[3]}**")
            col44.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[3], str(round(most_pos_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

    add_spacelines(2)

    with st.container():
        col11, col22, col33, col44 = st.columns(4)
        with col11:
            st.write(f"**{most_pos_users_names[4]}**")
            col11.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[4], str(round(most_pos_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

        with col22:
            st.write(f"**{most_pos_users_names[5]}**")
            col22.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[5], str(round(most_pos_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

        with col33:
            st.write(f"**{most_pos_users_names[6]}**")
            col33.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[6], str(round(most_pos_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

        with col44:
            st.write(f"**{most_pos_users_names[7]}**")
            col44.metric('Rhetoric behaviour score', most_pos_users['rhetoric_metric'].iloc[7], str(round(most_pos_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

        add_spacelines(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if radio_LEP_behavior == "target entities":
            pos_users_to_df = st.radio("Choose name to see details about the entity \n", most_pos_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            add_spacelines(1)
            st.write(f"sentences targeted at: **{pos_users_to_df}** ")
            st.dataframe(df[df.Target == str(pos_users_to_df)].set_index("source")[users_rhet_cols])
            add_spacelines(1)
        else:
            pos_users_to_df = st.radio("Choose name to see details about the user \n", most_pos_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"sentences posted by: **{pos_users_to_df}** ")
            st.dataframe(df[df.source == str(pos_users_to_df)].set_index("source")[users_rhet_cols])
            add_spacelines(1)

    user_stats_df_user2 = user_stats_df[user_stats_df['user'] == str(pos_users_to_df)]
    with st.container():
        if radio_LEP_behavior == "target entities":
            st.write(f"##### Users' rhetoric strategy to speak about {pos_users_to_df}")
        else:
            st.write(f"##### {pos_users_to_df}'s rhetoric strategy")
        add_spacelines(1)
        col222, col333 = st.columns(2)
        with col222:
            st.write(f"**Ethos strategy**")
            col222.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_support_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

        with col333:
            st.write(f"**Pathos strategy**")
            col333.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_positive_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

        strat_user_val_pos = [round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1),
                          round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1)]
        strat_user_val_pos_max = np.max(strat_user_val_pos)
        add_spacelines(1)
        strategy_lep_user_pos = []
        if radio_LEP_behavior == "target entities":
            if strat_user_val_pos[0] == strat_user_val_pos_max:
                strategy_lep_user_pos.append('ethos')
                #st.success(f"Users' positivity in speaking about **{pos_users_to_df}** comes mostly from **ethos**")
            if strat_user_val_pos[1] == strat_user_val_pos_max:
                strategy_lep_user_pos.append('pathos')
            strategy_lep_user_pos = " and ".join(strategy_lep_user_pos)
            st.success(f"Users' positivity in speaking about **{pos_users_to_df}** comes mostly from **{strategy_lep_user_pos}**")
        else:
            if strat_user_val_pos[0] == strat_user_val_pos_max:
                strategy_lep_user_pos.append('ethos')
                #st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **ethos**")
            if strat_user_val_pos[1] == strat_user_val_pos_max:
                strategy_lep_user_pos.append('pathos')
            strategy_lep_user_pos = " and ".join(strategy_lep_user_pos)
            st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **{strategy_lep_user_pos}**")
        add_spacelines(2)
        #st.write('<style>div.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)


def UserRhetStrategy(data):
    st.write(f" ### Rhetoric Strategies")
    df = data.copy()
    add_spacelines(1)
    plot_type_strategy = st.radio("Type of the plot", ('heatmap', 'histogram'))
    add_spacelines(1)

    rhetoric_dims = ['ethos', 'pathos']
    pathos_cols = ['pathos_label']

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n', 'pathos_n', 'pathos_negative_n', 'pathos_positive_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)
    cols_strat = ['ethos_support_percent', 'ethos_attack_percent',
                  'pathos_positive_percent', 'pathos_negative_percent']
    if plot_type_strategy == 'histogram':
        def plot_strategies(data):
            i = 0
            for c in range(2):
                sns.set(font_scale=1, style='whitegrid')
                print(cols_strat[c+i], cols_strat[c+i+1])
                fig_stats, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
                axs[0].hist(data[cols_strat[c+i]], color='#009C6F')
                title_str0 = " ".join(cols_strat[c+i].split("_")[:-1]).capitalize()
                axs[0].set_title(title_str0)
                axs[0].set_ylabel('number of users\n')
                axs[0].set_xlabel('\npercentage of texts %')
                axs[0].set_xticks(np.arange(0, 101, 10))

                axs[1].hist(data[cols_strat[c+i+1]], color='#9F0155')
                title_str1 = " ".join(cols_strat[c+i+1].split("_")[:-1]).capitalize()
                axs[1].set_xlabel('\npercentage of texts %')
                axs[1].yaxis.set_tick_params(labelbottom=True)
                axs[1].set_title(title_str1)
                axs[1].set_xticks(np.arange(0, 101, 10))
                plt.show()
                i+=1
                st.pyplot(fig_stats)
                add_spacelines(2)
        plot_strategies(data = user_stats_df)

    elif plot_type_strategy == 'heatmap':
        range_list = []
        number_users = []
        rhetoric_list = []
        bin_low = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        bin_high = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        dimensions = ['ethos_support_percent', 'pathos_positive_percent']
        for dim in dimensions:
            for val in zip(bin_low, bin_high):
                rhetoric_list.append(dim)
                range_list.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users.append(count_users)
        heat_df = pd.DataFrame({'range': range_list, 'values': number_users, 'dimension':rhetoric_list})
        heat_df['dimension'] = heat_df['dimension'].str.replace("_percent", "")
        heat_grouped = heat_df.pivot(index='range', columns='dimension', values='values')

        range_list_at = []
        number_users_at = []
        rhetoric_list_at = []
        dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent']
        for dim in dimensions_at:
            for val in zip(bin_low, bin_high):
                rhetoric_list_at.append(dim)
                range_list_at.append(str(val))
                count_users = len(user_stats_df[ (user_stats_df[dim] >= int(val[0])) & (user_stats_df[dim] <= int(val[1]))])
                number_users_at.append(count_users)
        heat_df_at = pd.DataFrame({'range': range_list_at, 'values': number_users_at, 'dimension':rhetoric_list_at})
        heat_df_at['dimension'] = heat_df_at['dimension'].str.replace("_percent", "")
        heat_grouped_at = heat_df_at.pivot(index='range', columns='dimension', values='values')

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.heatmap(heat_grouped_at, ax=axes[1], cmap='Reds', linewidths=0.1, annot=True)
        sns.heatmap(heat_grouped, ax=axes[0], cmap='Greens', linewidths=0.1, annot=True)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("range - percentage of texts %\n")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.show()
        st.pyplot(fig)
        add_spacelines(2)

    with st.container():
        ethos_strat = user_stats_df[(user_stats_df.ethos_percent > user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        pathos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent > user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean())]

        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            st.write('')
        with col2:
            st.write(f"Dominant **ethos** strategy ")
            col2.metric(str(ethos_strat.shape[0]) + " users", str(round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

        with col3:
            st.write(f"Dominant **pathos** strategy ")
            col3.metric(str(pathos_strat.shape[0]) + " users", str(round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

        #add_spacelines(2)
        #dominant_percent_strategy = round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)
        #col2.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant rhetoric strategy.")
        add_spacelines(2)





ethos_mapping = {0: 'neutral', 1: 'support', 2: 'attack'}
valence_mapping = {0: 'neutral', 1: 'positive', 2: 'negative'}




def TargetHeroScores_compare(data_list):
    st.write("## (Anti)Heroes Distribution")
    add_spacelines(2)

    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))
    add_spacelines(1)

    up_data_dict = {}
    up_data_dicth = {}
    up_data_dictah = {}
    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df["Target"] = df["Target"].apply(str)
        df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
        df["Target"] = df["Target"].str.replace('Government', 'government')

        if contents_radio_heroes == "direct ethos":
            targets_limit = df['Target'].unique()
            targets_limit = [t for t in targets_limit if "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()
        elif contents_radio_heroes == "3rd party ethos":
            targets_limit = df['Target'].unique()
            targets_limit = [t for t in targets_limit if not "@" in t]
            df = df[df.Target.isin(targets_limit)]
            if len(targets_limit) < 2:
                st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
                st.stop()

        dd2_size = df.groupby(['Target'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 1]
        adj_target = dd2_size['Target'].unique()

        dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
        dd.columns = ['normalized_value']
        dd = dd.reset_index()
        dd = dd[dd.Target.isin(adj_target)]
        dd = dd[dd.ethos_label != 'neutral']
        dd_hero = dd[dd.ethos_label == 'support']
        dd_antihero = dd[dd.ethos_label == 'attack']

        dd2 = pd.DataFrame({'Target': dd.Target.unique()})
        dd2_hist = dd2.copy()
        dd2anti_scores = []
        dd2hero_scores = []
        dd2['score'] = np.nan
        for t in dd.Target.unique():
            try:
                h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
            except:
                h = 0
            try:
                ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
            except:
                ah = 0
            dd2hero_scores.append(h)
            dd2anti_scores.append(ah)
            i = dd2[dd2.Target == t].index
            dd2.loc[i, 'score'] = h - ah

        dd2 = dd2[dd2.score != 0]
        dd2['ethos_label'] = np.where(dd2.score < 0, 'anti-hero', 'neutral')
        dd2['ethos_label'] = np.where(dd2.score > 0, 'hero', dd2['ethos_label'])
        dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
        dd2['score'] = dd2['score'] * 100
        #dd2['score'] = dd2[dd2['ethos_label'] != 'neutral' ]
        dd2_dist = pd.DataFrame(dd2['ethos_label'].value_counts(normalize=True).round(3)*100).reset_index()
        dd2_dist.columns = ['hero', 'percentage']
        dd2_dist['corpus'] = ds
        up_data_dict[n] = dd2_dist
        up_data_dicth[n] = dd2[dd2['ethos_label'] == 'hero']['Target'].unique()
        up_data_dictah[n] = dd2[dd2['ethos_label'] == 'anti-hero']['Target'].unique()
        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    sns.set(font_scale=1.65, style='whitegrid')
    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=5, aspect=1.2,
                    x = 'hero', y = 'percentage', hue = 'hero', dodge=False,
                    palette = {'anti-hero':'#BB0000', 'hero':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110), xlabel = '')
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points', fontsize=17)
    add_spacelines(2)
    with st.container():
        st.pyplot(f_dist_ethos)

        n_corp = len(data_list)
        add_spacelines(3)
        if n_corp == 2:
            plot1_comp_cnt, plot2_comp_cnt, = st.columns(2)
            with plot1_comp_cnt:
                i = 0
                plot1_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot1_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option1 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option1 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) > 3 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) > 3 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot2_comp_cnt:
                i = 1
                plot2_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot2_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)
                option2 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option2 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

        elif n_corp == 3:
            plot1_comp_cnt, plot2_comp_cnt, plot3_comp_cnt = st.columns(3)
            with plot1_comp_cnt:
                i = 0
                plot1_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot1_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)
                option1 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option1 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot2_comp_cnt:
                i = 1
                plot2_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot2_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option2 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option2 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot3_comp_cnt:
                i = 2
                plot3_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot3_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option3 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option3 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

        elif n_corp == 4:
            plot1_comp_cnt, plot2_comp_cnt, plot3_comp_cnt, plot4_comp_cnt = st.columns(4)
            with plot1_comp_cnt:
                i = 0
                plot1_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot1_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option1 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option1 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot2_comp_cnt:
                i = 1
                plot2_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot2_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option2 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option2 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot3_comp_cnt:
                i = 2
                plot3_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot3_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option3 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option3 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

            with plot4_comp_cnt:
                i = 3
                plot4_comp_cnt.metric(f"No. of **heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dicth[i]))
                add_spacelines(1)
                plot4_comp_cnt.metric(f"No. of **anti-heroes** in {data_list[i]['corpus'].iloc[0]}",
                            value = len(up_data_dictah[i]))
                add_spacelines(2)

                option4 = st.selectbox(f"See names of (anti)-heroes in **{data_list[i]['corpus'].iloc[0]}** ", ('Heroes', 'Anti-Heroes'))
                if option4 == 'Heroes':
                    names = up_data_dicth[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Greens')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dicth[i]}))

                else:
                    names = up_data_dictah[i]
                    names = [n if len(n) < 2 else "_".join(str(n).split()) for n in names]
                    f_att, words_att = make_word_cloud(" ".join(names), 700, 450, '#1E1E1E', 'Reds')
                    st.pyplot(f_att)
                    st.dataframe(pd.DataFrame({'name':up_data_dictah[i]}))

    add_spacelines(1)




def TargetHeroScores(data):
    st.write("## (Anti)Heroes")
    add_spacelines(1)
    df = data.copy()
    df["Target"] = df["Target"].apply(str)
    df = df[ (df.Target != 'nan') & (df.Target != '') & (df.ethos_label != 'neutral') ]
    df["Target"] = df["Target"].str.replace('Government', 'government')

    contents_radio_heroes = st.radio("Category of the target of ethotic statements", ("both", "direct ethos", "3rd party ethos"))
    add_spacelines(1)
    if contents_radio_heroes == "direct ethos":
        targets_limit = df['Target'].unique()
        targets_limit = [t for t in targets_limit if "@" in t]
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 2:
            st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
            st.stop()
    elif contents_radio_heroes == "3rd party ethos":
        targets_limit = df['Target'].unique()
        targets_limit = [t for t in targets_limit if not "@" in t]
        df = df[df.Target.isin(targets_limit)]
        if len(targets_limit) < 2:
            st.error(f'No cases of **{contents_radio_heroes}** found in the chosen corpora.')
            st.stop()

    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)

    dd2_size = df.groupby(['Target'], as_index=False).size()
    dd2_size = dd2_size[dd2_size['size'] > 1]
    adj_target = dd2_size['Target'].unique()

    dd = pd.DataFrame(df.groupby(['Target'])['ethos_label'].value_counts(normalize=True))
    dd.columns = ['normalized_value']
    dd = dd.reset_index()
    dd = dd[dd.Target.isin(adj_target)]
    dd = dd[dd.ethos_label != 'neutral']
    dd_hero = dd[dd.ethos_label == 'support']
    dd_antihero = dd[dd.ethos_label == 'attack']
    list_targets = dd["Target"].unique()
    list_targets = [x for x in list_targets if (str(x) != "nan" and len(str(x)) > 2) ]

    num_targets = len(list_targets)
    title1 = str(contents_radio)

    dd2 = pd.DataFrame({'Target': dd.Target.unique()})
    dd2_hist = dd2.copy()
    dd2anti_scores = []
    dd2hero_scores = []

    dd2['score'] = np.nan
    for t in dd.Target.unique():
        try:
            h = dd_hero[dd_hero.Target == t]['normalized_value'].iloc[0]
        except:
            h = 0
        try:
            ah = dd_antihero[dd_antihero.Target == t]['normalized_value'].iloc[0]
        except:
            ah = 0
        dd2hero_scores.append(h)
        dd2anti_scores.append(ah)
        i = dd2[dd2.Target == t].index
        dd2.loc[i, 'score'] = h - ah
    dd2['ethos_label'] = np.where(dd2.score < 0, 'anti-hero', 'neutral')
    dd2['ethos_label'] = np.where(dd2.score > 0, 'hero', dd2['ethos_label'])
    dd2 = dd2.sort_values(by = ['ethos_label', 'Target'])
    dd2['score'] = dd2['score'] * 100
    dd2['score'] = np.where(dd2.score == 0, 2, dd2['score'])
    dd2_hist['anti hero score'] = dd2anti_scores
    dd2_hist['hero score'] = dd2hero_scores
    dd2['corpus'] = str(title1)
    dd2_hist['corpus'] = str(title1)

    #st.write("#### (Anti)heroes distribution")
    with st.expander("(Anti)heroes distribution"):
        #color = sns.color_palette("Reds", 5)[-1:]  + sns.color_palette("Greens", 5)[::-1][:1] +  sns.color_palette("Blues", 5)[::-1][:1]
        color = {'anti-hero':'#BB0000', 'hero':'#026F00', 'neutral':'#022D96'}

        num_targets = int(len(list_targets))
        num_tar = int(num_targets / 3)
        if dd2.Target.nunique() > 50:
            len_targ = int(num_targets / 3)
            sns.set(font_scale=3, style='whitegrid')
            f1 = sns.catplot(kind = 'bar', data = dd2[dd2.Target.isin(list_targets[:int(num_tar)])].sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 20, alpha = 1, legend = False, col = "corpus")
            plt.ylabel("")
            f1.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.75, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()
            sns.set(font_scale=3, style='whitegrid')
            f2 = sns.catplot(kind = 'bar', data = dd2[dd2.Target.isin(list_targets[int(num_tar):int(num_tar)*2])].sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 20, alpha = 1, legend = False, col = "corpus")
            plt.ylabel("")
            f2.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.75, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()
            sns.set(font_scale=3, style='whitegrid')
            f3 = sns.catplot(kind = 'bar', data = dd2[dd2.Target.isin(list_targets[int(num_tar)*2:])].sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 20, alpha = 1, legend = False, col = "corpus")
            plt.ylabel("")
            f3.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.75, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()

            plot1, plot2, plot3 = st.columns(3)
            with plot1:
                st.pyplot(f1)
            with plot2:
                st.pyplot(f2)
            with plot3:
                st.pyplot(f3)

        elif dd2.Target.nunique() < 25:
            sns.set(font_scale=3, style='whitegrid')
            f1 = sns.catplot(kind = 'bar', data = dd2.sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 16, alpha = 1, legend = False, col = "corpus")
            plt.ylabel("")
            f1.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.75, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()
            with st.container():
                st.pyplot(f1)

        else:
            num_tar = int(num_targets / 2)
            sns.set(font_scale=3, style='whitegrid')
            f1 = sns.catplot(kind = 'bar', data = dd2[dd2.Target.isin(list_targets[:int(num_tar)])].sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 26, alpha = 1, legend = False, col = "corpus")

            plt.ylabel("")
            f1.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.73, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()
            sns.set(font_scale=3, style='whitegrid')
            f2 = sns.catplot(kind = 'bar', data = dd2[dd2.Target.isin(list_targets[int(num_tar):])].sort_values(by = ['Target', 'ethos_label']), y = 'Target', x = 'score',
                           hue = 'ethos_label', palette = color, dodge = False, sharey=False,
                           aspect = 1, height = 30, alpha = 1, legend = False, col = "corpus")
            plt.ylabel("")
            f1.set_axis_labels('\nscore', '')
            plt.legend(fontsize=30, title = '', bbox_to_anchor=(0.73, 1.13), ncol = 3)
            plt.tight_layout()
            plt.show()
            plot1, plot2 = st.columns(2)
            with plot1:
                st.pyplot(f1)
            with plot2:
                st.pyplot(f2)
    add_spacelines(2)

    #import random
    #rand_int = np.random.randint(0, int(len(list_targets)))

    selected_target = st.selectbox("Choose a target entity you would like to analyse", set(list_targets))

    # all df targets
    df_target_all = pd.DataFrame(df[df.ethos_label != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target_all.columns = ['percentage']
    df_target_all.reset_index(inplace=True)
    df_target_all.columns = ['label', 'percentage']
    df_target_all = df_target_all.sort_values(by = 'label')

    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

    # chosen target df
    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_label'].value_counts(normalize = True).round(2)*100)
    df_target.columns = ['percentage']
    df_target.reset_index(inplace=True)
    df_target.columns = ['label', 'percentage']

    if len(df_target) == 1:
      if not ("attack" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["attack", 0]
      elif not ("support" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["support", 0]

    df_target = df_target.sort_values(by = 'label')
    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]


    with st.container():
        st.info(f'Selected entity: ** {str(selected_target)} **')
        add_spacelines(1)
        col2, col1 = st.columns([3, 2])
        with col1:
            st.subheader("Hero score")
            col1.metric(str(selected_target), str(df_target_sup)+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'support') ])})" ,
            str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.'),
            help = f"Percentage (number in brackets) of texts that support ** {str(selected_target)} **") # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

        with col2:
            st.subheader("Anti-hero score")
            col2.metric(str(selected_target), str(df_target_att)+ str('%') + f" ({len(df[ (df.Target == str(selected_target)) & (df['ethos_label'] == 'attack') ])})",
            str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse",
            help = f"Percentage (number in brackets) of texts that attack ** {str(selected_target)} **") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

        add_spacelines(2)

        if not ("neutral" in df_target.label.unique()):
            df_target.loc[len(df_target)] = ["neutral", 0]
        df_target.columns = ['ethos', 'percentage']
        df_dist_ethos = df_target.sort_values(by = 'ethos')

        sns.set(font_scale=1, style='whitegrid')
        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4, aspect=1.4,
                        x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                        palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        plt.title(f"Ethos towards ** {str(selected_target)} **\n")
        plt.xlabel('')
        plt.yticks(np.arange(0, 105, 20), fontsize=13)
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))

        plot1, plot2, plot3 = st.columns([1, 6, 1])
        with plot2:
            st.pyplot(f_dist_ethos)

        #add_spacelines(2)
        #st.write('**********************************************************************************')
        #with st.expander("Target - valence analysis"):
            #add_spacelines(1)
            #sns.set(font_scale=1.5, style='whitegrid')
            #radio_senti_target = st.radio("Choose the unit of y-axis", ("percentage", "number"))

            #df_tar_emo_exp = df[df.Target == selected_target]
            #df_tar_emo_exp_senti = df_tar_emo_exp.groupby(['valence'], as_index=False).size()
            #df_tar_emo_exp_senti.sort_values(by = 'valence')
            #if radio_senti_target == "percentage":
                #df_tar_emo_exp_senti['size'] = round(df_tar_emo_exp_senti['size'] / len(df_tar_emo_exp), 3) * 100
            #df_tar_emo_exp_senti['valence'] = df_tar_emo_exp_senti['valence'].str.lower()

            #user_exp_labs = df_tar_emo_exp_senti['valence'].unique()
            #if not ('negative' in user_exp_labs):
                #df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['negative', 0]
            #if not ('neutral' in user_exp_labs):
            #    df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['neutral', 0]
            #if not ('positive' in user_exp_labs):
                #df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['positive', 0]

            #figsenti_user, axsenti = plt.subplots(figsize=(8, 5))
            #df_tar_emo_exp_senti = df_tar_emo_exp_senti.sort_values(by = 'valence')
            #axsenti.bar(df_tar_emo_exp_senti['valence'], df_tar_emo_exp_senti['size'], color = ['#BB0000', '#022D96', '#026F00'])
            #plt.xticks(fontsize=13)
            #plt.title(f"Valence towards ** {str(selected_target)} **\n", fontsize=15)
            #vals_senti = df_tar_emo_exp_senti['size'].values.round(1)
            #if radio_senti_target == "percentage":
                #plt.yticks(np.arange(0, 105, 20))
                #plt.ylabel('percentage %\n')
                #for index_senti, v in enumerate(vals_senti):
                    #plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))
            #else:
                #if len(df_tar_emo_exp) > 120:
                    #plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+16, 20), fontsize=12)
                #elif len(df_tar_emo_exp) > 40 and len(df_tar_emo_exp) < 120:
                    #plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+6, 5), fontsize=12)
                #else:
                #    plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+3, 2), fontsize=12)
                #plt.ylabel('number\n', fontsize=13)
                #for index_senti, v in enumerate(vals_senti):
                #    plt.text(x=index_senti , y = v , s=f"{v}" , fontdict=dict(fontsize=12, ha='center'))
            #plt.show()

            #col_val_hero1, col_val_hero2, col_val_hero3 = st.columns([1, 5, 1])
            #with col_val_hero1:
                #st.write('')
            #with col_val_hero3:
            #    st.write('')
            #with col_val_hero2:
            #    st.pyplot(figsenti_user)

        st.write('**********************************************************************************')
        #add_spacelines(1)
        cols = [
            'sentence', 'ethos_label', 'source', 'Target', 'pathos_label'] #, 'date', 'conversation_id'
        st.write('#### Cases of text supporting/attacking **', selected_target, ' **')
        if not "neutral" in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        st.dataframe(df[df.Target == str(selected_target)][cols].set_index('source').rename(columns={'ethos_label':'ethos'}), width = None)


def distribution_plot(data):
    df = data.copy()
    if not 'neutral' in df['ethos_label'].unique():
        df['ethos_label'] = df['ethos_label'].map(ethos_mapping)

    if not 'pathos_label' in df['pathos_label'].unique():
        df['pathos_label'] = df['pathos_label'].map(valence_mapping)
    df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)

    st.write("### Ethos distribution")
    add_spacelines(1)
    #with st.expander("Overall ethos distribution"):
    add_spacelines(2)
    df_dist_ethos.columns = ['percentage']
    df_dist_ethos.reset_index(inplace=True)
    df_dist_ethos.columns = ['ethos', 'percentage']
    df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')

    per = []
    eth = []
    eth.append('no ethos')
    per.append(float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    eth.append('ethos')
    per.append(100 - float(df_dist_ethos[df_dist_ethos.ethos == 'neutral']['percentage'].iloc[0]))
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':per})

    sns.set(font_scale=1.1, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'})
    f_dist_ethos0.set(ylim=(0, 110))
    plt.xlabel("")
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti0):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'})
    vals_senti = df_dist_ethos['percentage'].values.round(1)
    f_dist_ethos.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    for index_senti, v in enumerate(vals_senti):
        plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))


    df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)

    df_dist_ethos2.columns = ['percentage']
    df_dist_ethos2.reset_index(inplace=True)
    df_dist_ethos2.columns = ['ethos', 'percentage']
    df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')

    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'support':'#026F00'})
    f_dist_ethos2.set(ylim=(0, 110))
    plt.title(f"Ethos distribution in **{contents_radio}** \n")
    vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
    for index_senti2, v in enumerate(vals_senti2):
        plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

    plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
    with plot1_dist_ethos:
        st.write('')
    with plot2_dist_ethos:
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    with plot3_dist_ethos:
        st.write('')
    add_spacelines(2)


    with st.expander("Pathos distribution"):
        add_spacelines(1)
        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        df_dist_ethos.columns = ['pathos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')


        per = []
        eth = []
        eth.append('no pathos')
        per.append(float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        eth.append('pathos')
        per.append(100 - float(df_dist_ethos[df_dist_ethos.pathos == 'neutral']['percentage'].iloc[0]))
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':per})

        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'})
        f_dist_ethos0.set(ylim=(0, 110))
        plt.xlabel("")
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        vals_senti0 = df_dist_ethos_all0['percentage'].values.round(1)
        for index_senti2, v in enumerate(vals_senti0):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=13, ha='center'))

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#022D96', 'positive':'#026F00'})
        vals_senti = df_dist_ethos['percentage'].values.round(1)
        f_dist_ethos.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti, v in enumerate(vals_senti):
            plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))


        df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['pathos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')

        sns.set(font_scale=1.1, style='whitegrid')
        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos2, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00'})
        vals_senti2 = df_dist_ethos2['percentage'].values.round(1)
        f_dist_ethos2.set(ylim=(0, 110))
        plt.title(f"Pathos distribution in **{contents_radio}** \n")
        for index_senti2, v in enumerate(vals_senti2):
            plt.text(x=index_senti2, y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))

        plot1_dist_ethos, plot2_dist_ethos, plot3_dist_ethos = st.columns([1, 8, 1])
        with plot1_dist_ethos:
            st.write('')
        with plot2_dist_ethos:
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        with plot3_dist_ethos:
            st.write('')
        add_spacelines(1)




def TimeEthos_compare(data_list):
    st.write("### Compare ethos in time")
    add_spacelines(2)

    cols_columns = st.columns(len(data_list))
    dict_cond = {}
    for n, c in enumerate(cols_columns):
        with c:
            #df1 = data_list[n].copy()
            df = data_list[n].copy()
            df['key'] = + df.source + " -- " + df.Target
            df['key'] = np.where(df.ethos_label.isin(['attack', 'support', 2, 1]), df['key'], df['source'])
            df['support'] = np.where(df.ethos_label.isin([1, 'support']), 1, 0)
            df['attack'] = np.where(df.ethos_label.isin([2, 'attack']), 1, 0)
            st.write(f"Corpus: **{df.corpus.iloc[0]}** ")

            ######## new version - lineplot ##########
            dd = pd.DataFrame(df.groupby(['conversation_id']).size()).reset_index()
            dd2_size = df.groupby(['conversation_id'], as_index=False).size()
            dd2_size = dd2_size[dd2_size['size'] > 9]
            convs = dd2_size['conversation_id'].dropna().unique()

            dd.columns = ['conversation_id', 'size']
            df['support'] = np.where(df.ethos_label.isin([1, 'support']), 1, 0)
            df['attack'] = np.where(df.ethos_label.isin([2, 'attack']), 1, 0)
            convs_size_dict = {}
            for i in dd['size']:
                convs_size_dict[i] = dd[dd['size'] == i].conversation_id.to_list()

            max_size_cut = max(list(convs_size_dict.keys()))
            convs_turns_cut = [np.arange(10, 21, 1), np.arange(21, 36, 1),
                    np.arange(36, 56, 1),np.arange(56, 75, 1),np.arange(75, 101, 1),
                    np.arange(101, int(max_size_cut)+1, 1)]
                    #np.arange(101, 131, 1),np.arange(131, 161, 1), np.arange(161, 181, 1),

            for size_turns, turn_range in enumerate(convs_turns_cut):
                #print(size_turns, min(range), max(range))
                list_convs = [value for (key, value) in convs_size_dict.items() if key in convs_turns_cut[size_turns]]
                convs_cut = [v for l in list_convs for v in l]

                if len(convs_cut) > 0:
                    dict_convs_turns = {}
                    dict_convs_turns_s = {}
                    dict_convs_turns_a = {}
                    zz = 0
                    for con in convs_cut:
                      #print(con)
                      df1 = df[df.conversation_id == con]
                      if not 'time' in df1.columns:
                          df1['time'] = df1.full_text_id
                      if len(df1) > 1:
                        if len(set(df1.ethos_label.unique()).intersection(set([1, 2, 'support', 'attack']))) > 0:
                            df1 = df1.reset_index()
                            dict_convs_turns_s[zz] = df1[~(df1.ethos_label.isin([0, 'neutral']))]
                            df1 = df1.rename(columns={'index':'org_id'})
                            df1['key'] = np.where(df1.ethos_label.isin([1, 'support', 2, 'attack']), df1['key'], df1['source'])

                            df11 = df1.groupby(['time', 'org_id'], as_index=False)['support'].sum()
                            df12 = df1.groupby(['time', 'org_id'], as_index=False)['attack'].sum()

                            df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
                            df11 = df11.sort_values(by = ['time', 'org_id'], ascending=True).reset_index(drop=True).reset_index()

                            df11['seq'] = pd.cut(df11['index'], 3, labels=['beginning', 'middle', 'end'])
                            df11 = df11.groupby(['seq'], as_index=False)[['support', 'attack']].mean()# sum() mean()
                            dict_convs_turns[zz] = df11
                            zz += 1

                    df_ful_cut = dict_convs_turns[list(dict_convs_turns.keys())[0]].copy()
                    if len(list(dict_convs_turns.keys())) > 1:
                        for k in list(dict_convs_turns.keys())[1:]:
                            df_ful_cut[['support', 'attack']] = df_ful_cut.iloc[:, 1:]+dict_convs_turns[k].iloc[:, 1:]
                            df_ful_cut['support'] = np.mean([dict_convs_turns[k]['support'].values, df_ful_cut['support'].values], axis=0)
                            df_ful_cut['attack'] = np.mean([dict_convs_turns[k]['attack'].values, df_ful_cut['attack'].values], axis=0)

                    df11_melt = df_ful_cut[['seq', 'support', 'attack']].melt('seq', var_name='ethos')
                    sns.set(font_scale=1.5, style='whitegrid')
                    fig_line = sns.relplot(data = df11_melt, x = 'seq', y = 'value', hue='ethos',
                                kind='line', aspect=1.5, palette = {'support':'green', 'attack':'red'}, linewidth=2.8)
                    plt.xlabel('\ntime')
                    plt.ylabel('number')
                    plt.title(f"Range of sentences in discussion {min(turn_range)}-{max(turn_range)} ")
                    plt.show()

                    with st.container():
                        st.write(f'No. of diccussions: {len(convs_cut)} for range {min(turn_range)}-{max(turn_range)}.')
                        st.pyplot(fig_line)

    add_spacelines(2)
    cols_columns = st.columns(len(data_list))
    dict_cond = {}
    for n, c in enumerate(cols_columns):
        with c:
            #df1 = data_list[n].copy()
            df = data_list[n].copy()
            df['key'] = + df.source + " -- " + df.Target
            df['key'] = np.where(df.ethos_label.isin(['attack', 'support', 2, 1]), df['key'], df['source'])
            df['support'] = np.where(df.ethos_label.isin([1, 'support']), 1, 0)
            df['attack'] = np.where(df.ethos_label.isin([2, 'attack']), 1, 0)

            dict_convs_turns = {}
            zz = 0
            for con in df.conversation_id.unique():
              #print(con)
              df1 = df[df.conversation_id == con]
              if len(df1) > 1 and len(set(df1.ethos_label.unique()).intersection(set([1, 2, 'support', 'attack']))) > 0:
                  df1 = df1.reset_index()
                  if not 'time' in df1.columns:
                      df1['time'] = df1.full_text_id
                  df1 = df1.rename(columns={'index':'org_id'})
                  df11 = df1.groupby(['time', 'org_id'], as_index=False)['support'].sum()
                  df12 = df1.groupby(['time', 'org_id'], as_index=False)['attack'].sum()

                  df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
                  df11 = df11.sort_values(by = ['time', 'org_id'], ascending=True).reset_index(drop=True).reset_index()

                  df11['seq'] = pd.cut(df11['index'], 5, labels=['seq1', 'seq2', 'seq3', 'seq4', 'seq5'])
                  df11 = df11.groupby(['seq'], as_index=False)[['support', 'attack']].mean() # sum mean
                  dict_convs_turns[zz] = df11
                  zz += 1

            df_ful_cut = dict_convs_turns[list(dict_convs_turns.keys())[0]].copy()
            if len(list(dict_convs_turns.keys())) > 1:
                for k in list(dict_convs_turns.keys())[1:]:
                    #df_ful_cut[['support', 'attack']] = df_ful_cut.iloc[:, 1:]+dict_convs_turns[k].iloc[:, 1:]
                    df_ful_cut['support'] = np.mean([dict_convs_turns[k]['support'].values, df_ful_cut['support'].values], axis=0)
                    df_ful_cut['attack'] = np.mean([dict_convs_turns[k]['attack'].values, df_ful_cut['attack'].values], axis=0)

            df11_melt = df_ful_cut[['seq', 'support', 'attack']].melt('seq', var_name='ethos')
            sns.set(font_scale=1.5, style='whitegrid')
            fig_line = sns.relplot(data = df11_melt, x = 'seq', y = 'value', hue='ethos',
                        kind='line', aspect=1.5, palette = {'support':'green', 'attack':'red'}, linewidth=2.8)
            plt.xlabel('time')
            plt.ylabel('number')
            plt.title(f"Summary plot: general trend of ethos in time")
            plt.xticks(['seq1', 'seq2', 'seq3', 'seq4', 'seq5'],
                        labels = ['t1: 20%', 't2: 20-40%', 't3: 40-60%', 't4: 60-80%', 't5: 80-100%'])
            plt.show()
            with st.container():
                st.write('********************************************')
                st.write(f'No. of diccussions: {len(convs)} in **{df.corpus.iloc[0]}** corpus.')
                st.pyplot(fig_line)


def distribution_plot_compare(data_list):
    st.write("### Compare distributions")
    add_spacelines(2)

    up_data_dict = {}
    up_data_dict2 = {}
    n = 0
    for data in data_list:
        df = data.copy()
        ds = df['corpus'].iloc[0]
        #st.dataframe(df)
        if not 'attack' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        df_dist_ethos = pd.DataFrame(df['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos.columns = ['percentage']
        df_dist_ethos.reset_index(inplace=True)
        #st.dataframe(df_dist_ethos)
        df_dist_ethos.columns = ['ethos', 'percentage']
        df_dist_ethos = df_dist_ethos.sort_values(by = 'ethos')
        df_dist_ethos['corpus'] = ds
        up_data_dict[n] = df_dist_ethos

        df_dist_ethos2 = pd.DataFrame(df[df['ethos_label'] != 'neutral']['ethos_label'].value_counts(normalize = True).round(2)*100)
        df_dist_ethos2.columns = ['percentage']
        df_dist_ethos2.reset_index(inplace=True)
        df_dist_ethos2.columns = ['ethos', 'percentage']
        df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'ethos')
        df_dist_ethos2['corpus'] = ds
        up_data_dict2[n] = df_dist_ethos2

        n += 1

    df_dist_ethos_all = up_data_dict[0].copy()
    for k in range(int(len(up_data_dict.keys()))-1):
        k_sub = k+1
        df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

    neu = []
    eth = []
    corp = []
    for cor in df_dist_ethos_all.corpus.unique():
        corp.append(cor)
        nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['ethos'] == 'neutral')]['percentage'].iloc[0]
        neu.append(nn)
        eth.append('no ethos')
        neu.append(100 - nn)
        eth.append('ethos')
        corp.append(cor)
    df_dist_ethos_all0 = pd.DataFrame({'ethos':eth, 'percentage':neu, 'corpus':corp})

    sns.set(font_scale=1.5, style='whitegrid')
    f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'ethos':'#EA9200', 'no ethos':'#022D96'},
                    col = 'corpus')
    f_dist_ethos0.set(ylim=(0, 110))
    f_dist_ethos0.set(xlabel="")
    for ax in f_dist_ethos0.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'neutral':'#022D96', 'support':'#026F00'},
                    col = 'corpus')
    f_dist_ethos.set(ylim=(0, 110))
    for ax in f_dist_ethos.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    df_dist_ethos_all2 = up_data_dict2[0].copy()
    for k in range(int(len(up_data_dict2.keys()))-1):
        k_sub2 = k+1
        df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

    #st.dataframe(df_dist_ethos_all2)
    f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.4,
                    x = 'ethos', y = 'percentage', hue = 'ethos', dodge=False,
                    palette = {'attack':'#BB0000', 'support':'#026F00', 'neutral':'#022D96'},
                    col = 'corpus')

    f_dist_ethos2.set(ylim=(0, 110))
    for ax in f_dist_ethos2.axes.ravel():
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

    with st.container():
        st.pyplot(f_dist_ethos0)
        add_spacelines(1)
        st.pyplot(f_dist_ethos)
        add_spacelines(1)
        st.pyplot(f_dist_ethos2)
    add_spacelines(1)


    add_spacelines(1)
    with st.expander("Pathos distribution"):
        add_spacelines(1)
        up_data_dict = {}
        up_data_dict2 = {}
        n = 0
        for data in data_list:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            df = clean_text(df, 'sentence')
            df = lemmatization(df, 'content')
            if not 'negative' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)

            df_dist_ethos = pd.DataFrame(df['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos.columns = ['percentage']
            df_dist_ethos.reset_index(inplace=True)
            df_dist_ethos.columns = ['pathos', 'percentage']
            df_dist_ethos = df_dist_ethos.sort_values(by = 'pathos')
            up_data_dict[n] = df_dist_ethos
            df_dist_ethos['corpus'] = ds

            df_dist_ethos2 = pd.DataFrame(df[df['pathos_label'] != 'neutral']['pathos_label'].value_counts(normalize = True).round(2)*100)
            df_dist_ethos2.columns = ['percentage']
            df_dist_ethos2.reset_index(inplace=True)
            df_dist_ethos2.columns = ['pathos', 'percentage']
            df_dist_ethos2['corpus'] = ds
            df_dist_ethos2 = df_dist_ethos2.sort_values(by = 'pathos')
            up_data_dict2[n] = df_dist_ethos2

            n += 1
        df_dist_ethos_all = up_data_dict[0].copy()
        for k in range(int(len(up_data_dict.keys()))-1):
            k_sub = k+1
            df_dist_ethos_all = pd.concat([df_dist_ethos_all, up_data_dict[k_sub]], axis=0, ignore_index=True)

        neu = []
        eth = []
        corp = []
        for cor in df_dist_ethos_all.corpus.unique():
            corp.append(cor)
            nn = df_dist_ethos_all[(df_dist_ethos_all.corpus == cor) & (df_dist_ethos_all['pathos'] == 'neutral')]['percentage'].iloc[0]
            neu.append(nn)
            eth.append('no pathos')
            neu.append(100 - nn)
            eth.append('pathos')
            corp.append(cor)
        df_dist_ethos_all0 = pd.DataFrame({'pathos':eth, 'percentage':neu, 'corpus':corp})

        sns.set(font_scale=1.4, style='whitegrid')
        f_dist_ethos0 = sns.catplot(kind='bar', data = df_dist_ethos_all0, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'pathos':'#EA9200', 'no pathos':'#022D96'},
                        col = 'corpus')
        f_dist_ethos0.set(ylim=(0, 110))
        f_dist_ethos0.set(xlabel="")
        for ax in f_dist_ethos0.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        f_dist_ethos = sns.catplot(kind='bar', data = df_dist_ethos_all, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'neutral':'#022D96', 'positive':'#026F00'},
                        col = 'corpus')
        f_dist_ethos.set(ylim=(0, 110))
        for ax in f_dist_ethos.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        df_dist_ethos_all2 = up_data_dict2[0].copy()
        for k in range(int(len(up_data_dict2.keys()))-1):
            k_sub2 = k+1
            df_dist_ethos_all2 = pd.concat([df_dist_ethos_all2, up_data_dict2[k_sub2]], axis=0, ignore_index=True)

        f_dist_ethos2 = sns.catplot(kind='bar', data = df_dist_ethos_all2, height=4.5, aspect=1.4,
                        x = 'pathos', y = 'percentage', hue = 'pathos', dodge=False,
                        palette = {'negative':'#BB0000', 'positive':'#026F00', 'neutral':'#022D96'},
                        col = 'corpus')
        f_dist_ethos2.set(ylim=(0, 110))
        for ax in f_dist_ethos2.axes.ravel():
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.0f')+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

        with st.container():
            st.pyplot(f_dist_ethos0)
            add_spacelines(1)
            st.pyplot(f_dist_ethos)
            add_spacelines(1)
            st.pyplot(f_dist_ethos2)
        add_spacelines(1)




import time

##################### page config  #####################
st.set_page_config(page_title="EthAn", layout="wide") # centered wide

#####################  page content  #####################
st.title("EthAn - Ethos Analytics")
add_spacelines(1)


#  *********************** sidebar  *********************
with st.sidebar:
    st.title("Parameters of analysis")
    contents_radio_type = st.radio("Type of analysis", ('Single corpus', 'Compare corpora'))

    if contents_radio_type == 'Compare corpora':
        add_spacelines(1)
        contents_radio_type_compare = st.radio("Type of comparison", ('1 vs. 1', 'Group comparison'))
        add_spacelines(1)
        if contents_radio_type_compare == '1 vs. 1':
            st.write('Corpora')
            box_pol1 = st.checkbox("Covid-Vaccines Reddit")
            box_pol2 = st.checkbox("Covid-Vaccines Twitter", value=True)
            box_pol3 = st.checkbox("Climate-Change Reddit")
            box_pol4 = st.checkbox("Climate-Change Twitter", value=True)
            if not (int(box_pol1) + int(box_pol2) + int(box_pol4) + int(box_pol3) > 1):
                st.error('Choose at least 2 corpora')
                st.stop()
            corpora_list = []
            if box_pol1:
                cor11 = load_data(vac_red)
                cor1 = cor11.copy()
                cor1_src = cor1['source'].unique()
                cor1['conversation_id'] = 0
                cor1_src = [str(s).replace('@', '') for s in cor1_src]
                cor1['Target'] = cor1['Target'].apply(str)
                cor1['source'] = cor1['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                cor1['Target'] = cor1['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
                cor1['corpus'] = "Covid-Vaccines Reddit"
                cor1['time'] = cor1["original_id"].astype('int')
                cor1 = cor1.rename(columns = {'original_id':'full_text_id'})
                corpora_list.append(cor1)
            if box_pol2:
                cor22 = load_data(vac_tw)
                cor2 = cor22.copy()
                cor2_src = cor2['source'].unique()
                cor2_src = [str(s).replace('@', '') for s in cor2_src]
                cor2['Target'] = cor2['Target'].apply(str)
                cor2['Target'] = cor2['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor2_src) else x][0])
                cor2['corpus'] = "Covid-Vaccines Twitter"
                cor2['source'] = cor2['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                corpora_list.append(cor2)
            if box_pol3:
                cor33 = load_data(cch_red)
                cor3 = cor33.copy()
                cor3_src = cor3['source'].unique()
                cor3_src = [str(s).replace('@', '') for s in cor3_src]
                cor3['Target'] = cor3['Target'].apply(str)
                cor3['Target'] = cor3['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor3_src) else x][0])
                cor3['corpus'] = "Climate-Change Reddit"
                cor3['source'] = cor3['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                cor3['time'] = cor3["full_text_id"].astype('int')
                corpora_list.append(cor3)
            if box_pol4:
                cor44 = load_data(cch_tw)
                cor4 = cor44.copy()
                cor4_src = cor4['source'].unique()
                cor4_src = [str(s).replace('@', '') for s in cor4_src]
                cor4['Target'] = cor4['Target'].apply(str)
                cor4['Target'] = cor4['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor4_src) else x][0])
                cor4['corpus'] = "Climate-Change Twitter"
                cor4['source'] = cor4['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
                corpora_list.append(cor4)
        elif contents_radio_type_compare == 'Group comparison':
            corpora_list_names = ["Covid-Vaccines Reddit", "Covid-Vaccines Twitter", "Climate-Change Reddit", "Climate-Change Twitter"]
            corpora_paths = {"Covid-Vaccines Reddit": vac_red,
            "Covid-Vaccines Twitter": vac_tw,
            "Climate-Change Reddit": cch_red,
            "Climate-Change Twitter": cch_tw}

            options1 = st.multiselect('First group of corpora', corpora_list_names, corpora_list_names[:2])
            corpora_list_names_grp2 = set(corpora_list_names) - set(options1)
            corpora_list_names_grp2 = list(corpora_list_names_grp2)
            options2 = st.multiselect('Second group of corpora', corpora_list_names_grp2, corpora_list_names_grp2[0])

            shared_cols = ['sentence', 'source', 'Target', 'ethos_label', 'pathos_label',
            'corpus', 'full_text_id', 'conversation_id']
            data1 = load_data(corpora_paths[options1[0]])
            if not 'time' in data1.columns:
                if 'original_id' in data1.columns:
                    data1 = data1.rename(columns = {'original_id':'full_text_id'})
                    data1['conversation_id'] = 0
                data1['time'] = data1["full_text_id"].astype('int')
            data1['corpus'] = " &\n ".join(options1)
            data1 = data1#[shared_cols]
            if len(options1) > 1:
                for nn in range(int(len(options1))-1):
                    n = nn+1
                    data1_2 = load_data(corpora_paths[options1[int(n)]])
                    if not 'time' in data1_2.columns:
                        if 'original_id' in data1_2.columns:
                            data1_2 = data1_2.rename(columns = {'original_id':'full_text_id'})
                            data1_2['conversation_id'] = 0
                        data1_2['time'] = data1_2["full_text_id"].astype('int')
                    data1_2['corpus'] = " &\n ".join(options1)
                    data1_2 = data1_2#[shared_cols]
                    data1 = pd.concat( [data1, data1_2], axis=0, ignore_index=True )

            data2 = load_data(corpora_paths[options2[0]])
            data2['corpus'] = " &\n ".join(options2)
            if not 'time' in data2.columns:
                if 'original_id' in data2.columns:
                    data2 = data2.rename(columns = {'original_id':'full_text_id'})
                    data2['conversation_id'] = 0
                data2['time'] = data2["full_text_id"].astype('int')
            data2 = data2#[shared_cols]
            if len(options2) > 1:
                for nn in range(int(len(options2))-1):
                    n = nn+1
                    data2_2 = load_data(corpora_paths[options2[int(n)]])
                    if not 'time' in data2_2.columns:
                        if 'original_id' in data2_2.columns:
                            data2_2 = data2_2.rename(columns = {'original_id':'full_text_id'})
                            data2_2['conversation_id'] = 0
                        data2_2['time'] = data2_2["full_text_id"].astype('int')
                    data2_2['corpus'] = " &\n ".join(options2)
                    data2_2 = data2_2#[shared_cols]
                    data2 = pd.concat( [data2, data2_2], axis=0, ignore_index=True )

            corpora_list = []
            data1_src = data1['source'].unique()
            data1_src = [str(s).replace('@', '') for s in data1_src]
            data1['Target'] = data1['Target'].apply(str)
            data1['source'] = data1['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
            data1['Target'] = data1['Target'].apply(lambda x: "@"+str(x) if (not "@" in x and x in data1_src) else x)

            data2_src = data2['source'].unique()
            data2_src = [str(s).replace('@', '') for s in data2_src]
            data2['Target'] = data2['Target'].apply(str)
            data2['source'] = data2['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
            data2['Target'] = data2['Target'].apply(lambda x: "@"+str(x) if (not "@" in x and x in data2_src) else x)

            corpora_list.append(data1)
            corpora_list.append(data2)

        add_spacelines(1)
        contents_radio_an_cat = st.radio("Analytics Category", ('Text-based', 'Person-based', 'Time-based'))#, 'Time-based'
        add_spacelines(1)
        if contents_radio_an_cat == 'Person-based':
            contents_radio3 = st.radio("Analytics", {'(Anti)Heroes'})
        elif contents_radio_an_cat == 'Time-based':
            contents_radio3 = st.radio("Time", {'TimeAn'})
        else:
            contents_radio3 = st.radio("Analytics", ('Distribution', 'WordCloud'))
        add_spacelines(1)


    else:
        contents_radio = st.radio("Corpora", ("Covid-Vaccines Reddit",
                            "Covid-Vaccines Twitter", "Climate-Change Reddit", "Climate-Change Twitter"))
        add_spacelines(1)
        if contents_radio == "Covid-Vaccines Twitter":
            data = load_data(vac_tw)
            #df = data.copy()
            cor1_src = data['source'].unique()
            cor1_src = [str(s).replace('@', '') for s in cor1_src]
            data['Target'] = data['Target'].apply(str)
            data['Target'] = data['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
            data['corpus'] = str(contents_radio)
            data['source'] = data['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])

        elif contents_radio == "Covid-Vaccines Reddit":
            data = load_data(vac_red)
            #df = data.copy()
            data['time'] = data["original_id"].astype('int')
            data['conversation_id'] = 0
            data = data.rename(columns = {'original_id':'full_text_id'})
            cor1_src = data['source'].unique()
            cor1_src = [str(s).replace('@', '') for s in cor1_src]
            data['Target'] = data['Target'].apply(str)
            data['Target'] = data['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
            data['corpus'] = str(contents_radio)
            data['source'] = data['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])

        elif contents_radio == "Climate-Change Reddit":
            data = load_data(cch_red)
            #df = data.copy()
            cor1_src = data['source'].unique()
            cor1_src = [str(s).replace('@', '') for s in cor1_src]
            data['Target'] = data['Target'].apply(str)
            data['Target'] = data['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
            data['time'] = data.full_text_id
            data['source'] = data['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])
            #data = data.dropna(subset = ['comment_url'], axis=0)
            #data['comment_url'] = data['comment_url'].apply(str)
            #for i in data.index:
            #    newid = str(data.loc[i, 'comment_url']).split("/")
            #    if len(newid) > 5:
            #        data.loc[i, 'conversation_id'] = newid[4]
            #    else:
            #        data.loc[i, 'conversation_id'] = data.loc[i, 'conversation_id']
            data['corpus'] = str(contents_radio)

        elif contents_radio == "Climate-Change Twitter":
            data = load_data(cch_tw)
            #df = data.copy()
            cor1_src = data['source'].unique()
            cor1_src = [str(s).replace('@', '') for s in cor1_src]
            data['Target'] = data['Target'].apply(str)
            data['Target'] = data['Target'].apply(lambda x: ["@"+str(x) if (not "@" in x and x in cor1_src) else x][0])
            data['corpus'] = str(contents_radio)
            data['source'] = data['source'].apply(lambda x: ["@"+str(x) if not "@" in x else x][0])

        contents_radio_an_cat = st.radio("Analytics Category", ('Text-based', 'Person-based', 'Time-based'))
        add_spacelines(1)
        if contents_radio_an_cat == 'Person-based':
            contents_radio3 = st.radio("Analytics", ('(Anti)Heroes', "Rhetoric Strategies", "Rhetoric Profiles"))
        elif contents_radio_an_cat == 'Time-based':
            contents_radio3 = st.radio("Time", {'TimeAn'})
        else:
            contents_radio3 = st.radio("Analytics", ('Distribution', 'WordCloud', 'Explore copora'))
        add_spacelines(1)

        if (contents_radio3 == 'TimeAn') and (contents_radio == "Covid-Vaccines Reddit"):
            st.error('Time analysis for this corpora is not available.')
            st.stop()
            add_spacelines(2)

        #if contents_radio3 == 'TimeAn':
            #st.write("Category of analysis")
            #box_all = st.checkbox("All")
            #box_eth = st.checkbox("Ethos", value=True)
            #box_pat = st.checkbox("Pathos")
            #box_emoex = st.checkbox("Expressed emotions")
            #if not (box_eth or box_pat):
                #st.error('Choose a category for analysis')
                #assert_txt = st.button("Zatwierdź")
                #st.stop()


#####################  page content  #####################

if contents_radio_type == 'Compare corpora':
    if contents_radio3 == 'WordCloud':
        st.write('### Compare WordCloud')
        data_list = corpora_list.copy()

        add_spacelines(1)
        up_data_dict = {}
        texts_all_sup = []
        texts_all_att = []
        texts_all_pos = []
        texts_all_neg = []

        txt_sup = ''
        txt_att = ''
        txt_pos = ''
        txt_neg = ''

        n = 0

        for data in data_list:
            df = data.copy()
            ds = df['corpus'].iloc[0]

            df = clean_text(df, 'sentence')
            df = lemmatization(df, 'content')

            if not 'negative' in df['pathos_label'].unique():
                df['pathos_label'] = df['pathos_label'].map(valence_mapping)
            if not 'neutral' in df['ethos_label'].unique():
                df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
            up_data_dict[n] = df
            n += 1

            text1 = df[df.ethos_label == 'support']['content_lemmatized'].values
            text1 = " ".join(text1).lower().replace(' amp ', ' ').replace(' url ', ' ')
            #text1 = " ".join([w for w in text1.split() if text1.split().count(w) > 1])
            txt_sup += text1
            text1 = set(text1.split())
            texts_all_sup.append(text1)
            text2 = df[df.ethos_label == 'attack']['content_lemmatized'].values
            text2 = " ".join(text2).lower().replace(' amp ', ' ').replace(' url ', ' ')
            #text2 = " ".join([w for w in text2.split() if text2.split().count(w) > 1])
            txt_att += text2
            text2 = set(text2.split())
            texts_all_att.append(text2)

            text11 = df[df.pathos_label == 'positive']['content_lemmatized'].values
            text11 = " ".join(text11).lower().replace(' amp ', ' ').replace(' url ', ' ')
            #text11 = " ".join([w for w in text11.split() if text11.split().count(w) > 1])
            txt_pos += text11
            text11 = set(text11.split())
            texts_all_pos.append(text11)
            text22 = df[df.pathos_label == 'negative']['content_lemmatized'].values
            text22 = " ".join(text22).lower().replace(' amp ', ' ').replace(' url ', ' ')
            #text22 = " ".join([w for w in text22.split() if text22.split().count(w) > 1])
            txt_neg += text22
            text22 = set(text22.split())
            texts_all_neg.append(text22)

        #shared ethos
        shared_all_sup = []
        shared_all_att = []
        shared_all_pos = []
        shared_all_neg = []

        shared_sup = texts_all_sup[0]
        shared_att = texts_all_att[0]
        shared_pos = texts_all_pos[0]
        shared_neg = texts_all_neg[0]

        for n in range(int(len(data_list))-1):
            shared1 = shared_sup.intersection(texts_all_sup[n+1])
            shared_all_sup.extend(list(shared1))

            shared11 = shared_att.intersection(texts_all_att[n+1])
            shared_all_att.extend(list(shared11))

            shared2 = shared_pos.intersection(texts_all_pos[n+1])
            shared_all_pos.extend(list(shared2))

            shared22 = shared_neg.intersection(texts_all_neg[n+1])
            shared_all_neg.extend(list(shared22))


        textatt_a = " ".join(shared_all_att)
        for word11 in shared_all_att:
            word_cnt = txt_att.split().count(str(word11))
            word_cnt_l = " ".join([str(word11)] * int(word_cnt))
            textatt_a += word_cnt_l
        textatt_a = textatt_a.split()
        textatt_a = [w for w in textatt_a if txt_att.split().count(w) > 4]
        random.shuffle(textatt_a)
        if len(textatt_a) < 1:
            textatt_a = ['empty', 'empty']
            textatt_a = " ".join(textatt_a)
        elif len(textatt_a) == 1:
            textatt_a = str(textatt_a[0])
        else:
            textatt_a = " ".join(textatt_a)

        textpos_a = " ".join(shared_all_pos)
        for word2 in shared_all_pos:
            word_cnt = txt_pos.split().count(str(word2))
            word_cnt_l = " ".join([str(word2)] * int(word_cnt))
            textpos_a += word_cnt_l
        textpos_a = textpos_a.split()
        textpos_a = [w for w in textpos_a if txt_pos.split().count(w) > 4]
        random.shuffle(textpos_a)
        if len(textpos_a) < 1:
            textpos_a = ['empty', 'empty']
            textpos_a = " ".join(textpos_a)
        elif len(textpos_a) == 1:
            textpos_a = str(textpos_a[0])
        else:
            textpos_a = " ".join(textpos_a)

        textneg_a = " ".join(shared_all_neg)
        for word in shared_all_neg:
            word_cnt = txt_neg.split().count(str(word))
            word_cnt_l = " ".join([str(word)]* int(word_cnt))
            textneg_a += word_cnt_l
        textneg_a = textneg_a.split()
        textneg_a = [w for w in textneg_a if txt_neg.split().count(w) > 4]
        random.shuffle(textneg_a)
        if len(textneg_a) < 1:
            textneg_a = ['empty', 'empty']
            textneg_a = " ".join(textneg_a)
        elif len(textneg_a) == 1:
            textneg_a = str(textneg_a[0])
        else:
            textneg_a = " ".join(textneg_a)

        textsup_a = shared_all_sup.copy()
        for word in shared_all_sup:
            word_cnt = txt_sup.split().count(str(word))
            word_cnt_l = [str(word)] * int(word_cnt)
            textsup_a.extend(word_cnt_l)
        #textsup_a = textsup_a.split()
        textsup_a = [w for w in textsup_a if txt_sup.split().count(w) > 4]
        #random.shuffle(textsup_a)
        if len(textsup_a) < 1:
            textsup_a = ['empty', 'empty']
            textsup_a = " ".join(textsup_a)
        elif len(textsup_a) == 1:
            if len(str(textsup_a[0])) > 15:
                textsup_a = str(textsup_a[0])[:15]
            else:
                textsup_a = str(textsup_a[0])
        else:
            textsup_a = " ".join(textsup_a)


        f_sup, words_sup = make_word_cloud(textsup_a, 800, 500, '#1E1E1E', 'Greens')
        f_att, words_att = make_word_cloud(textatt_a, 800, 500, '#1E1E1E', 'Reds')
        f_pos, words_pos = make_word_cloud(textpos_a, 800, 500, '#1E1E1E', 'Greens')
        f_neg, words_neg = make_word_cloud(textneg_a, 800, 500, '#1E1E1E', 'Reds')

        col1_cl_et, col2_cl_et = st.columns(2, gap='large')# [5, 4]
        with col1_cl_et:
            st.write('Shared **ethos support** words')
            st.pyplot(f_sup)
            st.write(f"There are {len(set(textsup_a.split()))} shared words.")

            add_spacelines(1)
            st.write('Shared **pathos positive** words')
            st.pyplot(f_pos)
            st.write(f"There are {len(set(textpos_a.split()))} shared words.")

        with col2_cl_et:
            st.write('Shared **ethos attack** words')
            st.pyplot(f_att)
            st.write(f"There are {len(set(textatt_a.split()))} shared words.")

            add_spacelines(1)
            st.write('Shared **pathos negative** words')
            st.pyplot(f_neg)
            st.write(f"There are {len(set(textneg_a.split()))} shared words.")
        add_spacelines(2)

        st.write("#### Cases")
        add_spacelines(1)
        exp_data = up_data_dict[0]
        for n in range(int(len(data_list))-1):
            exp_data = pd.concat([exp_data, up_data_dict[n+1]], axis=0, ignore_index=True)

#        exp_cat = st.selectbox("Category of examples", ("ethos support", "ethos attack", "valence positive", "valence negative"))
        look_dict_exp = {
        "ethos support":textsup_a, "ethos attack":textatt_a,
        "pathos positive":textpos_a, "pathos negative":textneg_a}

        for exp_cat in ["ethos support", "ethos attack", "pathos positive", "pathos negative"]:
            with st.expander(exp_cat):
                cols_all = ['sentence', 'source', 'Target', 'ethos_label', 'pathos_label', 'corpus']
                c1 = f"shared_{exp_cat.split()[-1]}_words"
                c1 = str(c1)
                c2 = f"shared_{exp_cat.split()[-1]}"
                c2 = str(c2)
                cols_all.append(c1)
                cols_all.append(c2)

                exp_sup = exp_data.copy()
                exp_sup[c1] = exp_sup.content_lemmatized.apply(lambda x: set(x.split()).intersection(set(look_dict_exp[exp_cat].split())))
                exp_sup[c2] = exp_sup[c1].map(len)
                exp_sup = exp_sup[exp_sup[c2] > 0]

                st.dataframe(exp_sup[cols_all], width = None)
        st.stop()

    elif contents_radio3 == '(Anti)Heroes':
        TargetHeroScores_compare(data_list = corpora_list)
    elif contents_radio3 == 'TimeAn':
        TimeEthos_compare(data_list = corpora_list)
    else:
        distribution_plot_compare(data_list = corpora_list)

else:
    if not contents_radio in ["Covid-Vaccines Reddit"]:
        df = data.copy()
        convs = df.conversation_id.dropna().unique()
    if not contents_radio in ["Covid-Vaccines Reddit", "Climate-Change Reddit"]:
        df = data.copy()
        df = df[df.conversation_id.isin(convs)]
        df['date'] = df['date'].fillna(method='ffill')
        df['time'] = pd.to_datetime(df.date).apply(str)
        df['Target'] = df['Target'].fillna('')
        df['key'] = + df.source + " -- " + df.Target
        df['key'] = np.where(df.ethos_label != 0, df['key'], df['source'])
        df = df.drop_duplicates(['full_text', 'sentence'])

    else:
        df = data.copy()
        df['Target'] = df['Target'].fillna('')
        df['key'] = + df.source + " -- " + df.Target
        df['key'] = np.where(df.ethos_label != 0, df['key'], df['source'])
        df = df.drop_duplicates(['sentence'])



    ###########  Pre-process ##############
    df = clean_text(df, 'sentence')
    df = lemmatization(df, 'content')

    if contents_radio3 == "(Anti)Heroes":
        TargetHeroScores(data = df)

    elif contents_radio3 == 'Distribution':
        distribution_plot(data = df)

    elif contents_radio3 == 'Rhetoric Strategies':
        UserRhetStrategy(data = df)

    elif contents_radio3 == 'Rhetoric Profiles':
        UsersExtreme(data = df)



    elif contents_radio3 == "WordCloud":
        st.write('### WordCloud')
        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)

        col1_cl, col2_cl = st.columns(2, gap='large')# [5, 4]
        with col1_cl:
            st.write('**Ethos support**')
            text1 = df[df.ethos_label == 'support']['content_lemmatized'].values
            text1 = " ".join(text1).lower().replace(' amp ', ' ').replace(' url ', ' ')
            fig_cl1, words_ethos_wcl = make_word_cloud(text1, 1000, 620, '#1E1E1E', 'Greens') #gist_heat / flare_r crest viridis
            st.pyplot(fig_cl1)

            add_spacelines(1)
            st.write('**Pathos positive**')
            text11 = df[df.pathos_label == 'positive']['content_lemmatized'].values
            text11 = " ".join(text11).lower().replace(' amp ', ' ').replace(' url ', ' ').replace(' url ', ' ')
            fig_cl11, words_pathos_wcl = make_word_cloud(text11, 1000, 620, '#1E1E1E', 'Greens')
            st.pyplot(fig_cl11)

        text2 = df[df.ethos_label == 'attack']['content_lemmatized'].values
        text2 = " ".join(text2).lower().replace(' amp ', ' ').replace(' url ', ' ').replace(' url ', ' ')
        fig_cl2, words_ethos_wcl_neg = make_word_cloud(text2, 1000, 620, '#1E1E1E', 'Reds') #gist_heat / flare_r crest viridis
        with col2_cl:
            st.write('**Ethos attack**')
            st.pyplot(fig_cl2)
            add_spacelines(1)
            st.write('**Pathos negative**')
            text22 = df[df.pathos_label == 'negative']['content_lemmatized'].values
            text22 = " ".join(text22).lower().replace(' amp ', ' ').replace(' url ', ' ')
            fig_cl22, words_pathos_wcl_neg = make_word_cloud(text22, 1000, 620, '#1E1E1E', 'Reds')
            st.pyplot(fig_cl22)
        add_spacelines(2)

        st.write('Words in **ethos support and pathos positive**:')
        st.write(pd.DataFrame(list(set(list(words_ethos_wcl)).intersection(set(list(words_pathos_wcl))))).T)
        st.write('Words in **ethos attack and pathos negative**:')
        st.write(pd.DataFrame(list(set(list(words_ethos_wcl_neg)).intersection(set(list(words_pathos_wcl_neg))))).T)

        st.write('Words in **ethos support and attack**:')
        st.write(pd.DataFrame(list(set(list(words_ethos_wcl)).intersection(set(list(words_ethos_wcl_neg))))).T)
        st.write('Words in **pathos positive and negative**:')
        st.write(pd.DataFrame(list(set(list(words_pathos_wcl)).intersection(set(list(words_pathos_wcl_neg))))).T)


    elif contents_radio3 == 'Explore copora':
        #st.dataframe(df)
        st.write('### Explore copora')
        dff_columns = ['sentence', 'source', 'Target', 'ethos_label', 'pathos_label']# , 'conversation_id','date'

        if not 'neutral' in df['ethos_label'].unique():
            df['ethos_label'] = df['ethos_label'].map(ethos_mapping)
        if not 'neutral' in df['pathos_label'].unique():
            df['pathos_label'] = df['pathos_label'].map(valence_mapping)
        dff = df.copy()
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[-2:])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        st.dataframe(dff_selected[dff_columns].set_index("source"), width = None)
        add_spacelines(1)
        st.write(f"No. of cases: {len(dff_selected)}.")
        #st.dataframe(df_exp[dff_columns])


    else:
        st.write("## Analysis of ethos in time")
        add_spacelines(2)

        sns.set(font_scale=1.2, style='whitegrid')

        st.write("**Processing the output ...**")
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        add_spacelines(2)

        st.write("### Lineplots")
        df['full_text'] = df['full_text'].str.replace('> ', '').str.replace('&amp; ', '')
        df1 = df.copy()

        ######## new version - lineplot ##########
        dd = pd.DataFrame(df.groupby(['conversation_id']).size()).reset_index()
        dd2_size = df.groupby(['conversation_id'], as_index=False).size()
        dd2_size = dd2_size[dd2_size['size'] > 5]
        convs = dd2_size['conversation_id'].dropna().unique()

        dd.columns = ['conversation_id', 'size']
        df['support'] = np.where(df.ethos_label.isin([1, 'support']), 1, 0)
        df['attack'] = np.where(df.ethos_label.isin([2, 'attack']), 1, 0)
        convs_size_dict = {}
        for i in dd['size']:
            convs_size_dict[i] = dd[dd['size'] == i].conversation_id.to_list()

        max_size_cut = max(list(convs_size_dict.keys()))
        convs_turns_cut = [np.arange(5, 15, 1), np.arange(15, 26, 1),np.arange(26, 31, 1),
                np.arange(31, 46, 1),np.arange(46, 61, 1),
                np.arange(61, 86, 1),np.arange(86, 101, 1),
                np.arange(101, 131, 1),np.arange(131, 161, 1), np.arange(161, 181, 1), np.arange(181, int(max_size_cut)+1, 1)]


        for size_turns, turn_range in enumerate(convs_turns_cut):
            #print(size_turns, min(range), max(range))
            list_convs = [value for (key, value) in convs_size_dict.items() if key in convs_turns_cut[size_turns]]
            convs_cut = [v for l in list_convs for v in l]

            if len(convs_cut) > 0:
                dict_convs_turns = {}
                zz = 0
                for con in convs_cut:
                  #print(con)
                  df1 = df[df.conversation_id == con]
                  if not 'time' in df1.columns:
                      df1['time'] = df1.full_text_id
                  if len(df1) > 1:
                    df1 = df1.reset_index()
                    df1 = df1.rename(columns={'index':'org_id'})
                    df1['key'] = np.where(df1.ethos_label.isin([1, 'support', 2, 'attack']), df1['key'], df1['source'])

                    df11 = df1.groupby(['time', 'org_id'], as_index=False)['support'].sum()
                    df12 = df1.groupby(['time', 'org_id'], as_index=False)['attack'].sum()

                    df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
                    df11 = df11.sort_values(by = ['time', 'org_id'], ascending=True).reset_index(drop=True).reset_index()

                    df11['seq'] = pd.cut(df11['index'], 3, labels=['beginning', 'middle', 'end'])
                    df11 = df11.groupby(['seq'], as_index=False)[['support', 'attack']].sum()
                    dict_convs_turns[zz] = df11
                    zz += 1

                df_ful_cut = dict_convs_turns[list(dict_convs_turns.keys())[0]].copy()
                if len(list(dict_convs_turns.keys())) > 1:
                    for k in list(dict_convs_turns.keys())[1:]:
                        #print(k)
                        df_ful_cut[['support', 'attack']] = df_ful_cut.iloc[:, 1:]+dict_convs_turns[k].iloc[:, 1:]

                df11_melt = df_ful_cut[['seq', 'support', 'attack']].melt('seq', var_name='ethos')

                fig_line = sns.relplot(data = df11_melt, x = 'seq', y = 'value', hue='ethos',
                            kind='line', aspect=1.5, palette = {'support':'green', 'attack':'red'})
                plt.xlabel('\ntime')
                plt.ylabel('number')
                plt.title(f"Range of sentences in discussion {min(turn_range)}-{max(turn_range)} ")
                plt.show()

                c1_line, c2_line, c3_line = st.columns([1, 6, 1])
                with c1_line:
                    st.write('')
                with c3_line:
                    st.write('')
                with c2_line:
                    st.write(f'#####  No. of diccussions: {len(convs_cut)} for range {min(turn_range)}-{max(turn_range)}.')
                    st.pyplot(fig_line)

        dict_convs_turns = {}
        zz = 0
        for con in df.conversation_id.unique():
          #print(con)
          df1 = df[df.conversation_id == con]
          if len(df1) > 1:
            df1 = df1.reset_index()
            if not 'time' in df1.columns:
                df1['time'] = df1.full_text_id
            df1 = df1.rename(columns={'index':'org_id'})
            df1['key'] = np.where(df1.ethos_label != 0, df1['key'], df1['source'])

            df11 = df1.groupby(['time', 'org_id'], as_index=False)['support'].sum()
            df12 = df1.groupby(['time', 'org_id'], as_index=False)['attack'].sum()

            df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
            df11 = df11.sort_values(by = ['time', 'org_id'], ascending=True).reset_index(drop=True).reset_index()

            df11['seq'] = pd.cut(df11['index'], 5, labels=['seq1', 'seq2', 'seq3', 'seq4', 'seq5'])
            df11 = df11.groupby(['seq'], as_index=False)[['support', 'attack']].sum()
            dict_convs_turns[zz] = df11
            zz += 1

        df_ful_cut = dict_convs_turns[list(dict_convs_turns.keys())[0]].copy()
        if len(list(dict_convs_turns.keys())) > 1:
            for k in list(dict_convs_turns.keys())[1:]:
                #print(k)
                df_ful_cut[['support', 'attack']] = df_ful_cut.iloc[:, 1:]+dict_convs_turns[k].iloc[:, 1:]

        df11_melt = df_ful_cut[['seq', 'support', 'attack']].melt('seq', var_name='ethos')

        fig_line = sns.relplot(data = df11_melt, x = 'seq', y = 'value', hue='ethos',
                    kind='line', aspect=1.5, palette = {'support':'green', 'attack':'red'})
        plt.xlabel('time')
        plt.ylabel('number')
        plt.title(f"Summary plot: general trend of ethos in time")
        plt.xticks(['seq1', 'seq2', 'seq3', 'seq4', 'seq5'],
                    labels = ['seq1: 20%', 'seq2: 20-40%', 'seq3: 40-60%', 'seq4: 60-80%', 'seq5: 80-100%'])
        plt.show()

        c1_line, c2_line, c3_line = st.columns([1, 6, 1])
        with c1_line:
            st.write('')
        with c3_line:
            st.write('')
        with c2_line:
            st.write('********************************************')
            st.write(f'#####  No. of diccussions: {len(convs)} in {contents_radio} corpus.')
            st.pyplot(fig_line)

        add_spacelines(2)

        st.write("### Detailed analysis")
        with st.expander("Expand"):

            col1_exp, col2_exp, col3_exp = st.columns([1, 3, 1], gap="medium")
            with col1_exp:
                st.write("")
            with col3_exp:
                st.write("")
            with col2_exp:
                for con in convs[::8]:
                    if not 0 in df.ethos_label.unique():
                        df['ethos_label'] = df['ethos_label'].map({'support':1, 'neutral':0, 'attack':2})
                    df1 = df[df.conversation_id == con]
                    df1 = df1.reset_index()
                    if not 'time' in df1.columns:
                        df1['time'] = df1.full_text_id
                    df1 = df1.rename(columns={'index':'org_id'})
                    df11 = df1.groupby(['time', 'org_id'], as_index=False)['key'].max()
                    df12 = df1.groupby(['time','org_id'], as_index=False)['ethos_label'].max()
                    df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
                    df11 = df11.sort_values(by = 'time', ascending=False).reset_index(drop=True).reset_index()
                    df11['ethos_label'] = df11['ethos_label'].map({1:2, 0:1, 2:0})

                    if len(df11) > 50:
                        fig, ax = plt.subplots(figsize=(8, 33))
                    elif len(df11) > 37:
                        fig, ax = plt.subplots(figsize=(8, 26))
                    elif len(df11) > 25:
                        fig, ax = plt.subplots(figsize=(8, 22))
                    else:
                        fig, ax = plt.subplots(figsize=(6, 7))

                    sns.set(font_scale=1.1, style='whitegrid')
                    ax.plot(df11['ethos_label'], df11['index'], color = 'grey', alpha=0.65, linewidth=2.2)
                    sns.scatterplot(data = df11, y = 'index', x = 'ethos_label', hue='ethos_label', ax=ax,
                                    palette = ({1:'blue', 0: 'red', 2:'green'}), s=170, legend=False)#,

                    yy = df11['index'].values
                    xx = df11.ethos_label.values
                    ss = df11.key.values
                    for i, v in enumerate(xx):
                        if len(xx) > 5:
                            plt.text(xx[i]+0.1, yy[i]-0.15, ss[i], fontsize = 10, weight='bold')#, rotation=90
                        else:
                            plt.text(xx[i]+0.1, yy[i], ss[i], fontsize = 10, weight='bold')
                    plt.xlim(-1, 3.5)

                    plt.yticks( np.arange(0, df11['index'].iloc[-1]+1, 1), df11['time'].values, fontsize=8)
                    import matplotlib.patches as mpatches
                    r_patch = mpatches.Patch(color='red', label='attack')
                    b_patch = mpatches.Patch(color='blue', label='no ethos')
                    g_patch = mpatches.Patch(color='green', label='support')
                    plt.legend(handles = [r_patch, b_patch, g_patch], fontsize=10, title = 'ethos',
                    bbox_to_anchor=(1.24, 0.96), ncol = 1)
                    plt.xticks([], [])
                    #if ('pathos_label' in df1.columns) and (box_pat):
                        #if not 0 in df1.pathos_label.unique():
                            #df1['ethos_label'] = df1['pathos_label'].map({'positive':1, 'neutral':0, 'negative':2})
                        #df11 = df1.groupby(['time', 'org_id'], as_index=False)['key'].max()
                        #df12 = df1.groupby(['time','org_id'], as_index=False)['pathos_label'].max()
                        #df11 = pd.concat( [df11, df12.iloc[:, -1]], axis=1 )
                        #df11 = df11.sort_values(by = 'time', ascending=False).reset_index(drop=True).reset_index()
                        #df11['pathos_label'] = df11['pathos_label'].map({1:0.9, 2:1.85, 0:0})
                        #sns.scatterplot(data = df11, y = 'index', x = 'pathos_label', hue='pathos_label',
                        #                  palette = ({1.85: 'red', 0.9:'green', 0:'blue'}),
                        #                  s=450, ax=ax, legend=False, marker = '*', alpha=0.7)

                    plt.ylabel('time\n < --------------------- ')
                    plt.xlabel('')
                    if 'date' in df.columns:
                        plt.title('Id: '+str(con)+'\n'+'@'+df[df.conversation_id == con].sort_values(by = 'date')['source'].iloc[0]+\
                                ': '+df[df.conversation_id == con].sort_values(by = 'date')['full_text'].iloc[0].replace('\n', ' ')[:55]+'\n'+df[df.conversation_id == con].sort_values(by = 'date')['full_text'].iloc[0].replace('\n', ' ')[55:130] + '...')
                    else:
                        plt.title('Id: '+str(con)+'\n'+'@'+df[df.conversation_id == con].sort_values(by = 'time')['source'].iloc[0]+\
                                ': '+df[df.conversation_id == con].sort_values(by = 'time')['full_text'].iloc[0].replace('\n', ' ')[:55]+'\n'+df[df.conversation_id == con].sort_values(by = 'time')['full_text'].iloc[0].replace('\n', ' ')[55:130] + '...')
                    plt.show()
                    st.pyplot(fig)

        add_spacelines()
        with st.container():
            df1 = df.copy()
            df1[['support', 'attack']] = df1[['support', 'attack']].fillna(0)

            st.write('### Correlation analysis')
            add_spacelines()
            col1_corr, col2_corr = st.columns(2, gap='large')# [5, 4]
            with col1_corr:
                # support
                con_id = []
                att = []
                for cid in df1.conversation_id.unique():
                    if 1 in df1[df1.conversation_id == cid].support.unique():
                        df11 = df1[df1.conversation_id == cid].groupby(['time'], as_index=False)['support'].max()
                        con_id.extend( list(df11.index) )
                        att.extend(df11.support.to_list())
                coef2, pval2 = pointbiserialr(con_id, att)

                if pval2 >= 0.05:
                    col1_corr.metric("Correlation between ethotic supports and time",
                                value = round(coef2, 3), delta = round(pval2, 4), delta_color="inverse")
                elif pval2 < 0.05:
                    col1_corr.metric("Correlation between ethotic supports and time",
                                value = round(coef2, 3), delta = round(pval2, 4))

                import scipy.stats as stats
                part1_sup = 0
                part1_neu = 0
                part2_sup = 0
                part2_neu = 0
                for cid in df1.conversation_id.unique():
                    if 1 in df1[df1.conversation_id == cid].support.unique():
                        partition = int(len(df1[df1.conversation_id == cid])/2)
                        part1 = df1[df1.conversation_id == cid].iloc[:partition]
                        part2 = df1[df1.conversation_id == cid].iloc[partition:]
                        part1_sup += part1.support.sum()
                        part2_sup += part2.support.sum()
                        part1_neu += part1.shape[0] - part1.support.sum()
                        part2_neu += part2.shape[0] - part2.support.sum()

                data = np.array([[part1_neu, part1_sup], [part2_neu, part2_sup]])
                #Chi-squared test statistic, sample size, and minimum of rows and columns
                X2 = stats.chi2_contingency(data, correction=False)[0]
                n = np.sum(data)
                minDim = min(data.shape)-1
                #calculate Cramer's V
                V = np.sqrt((X2/n) / minDim)
                #st.write(f"Cramer's V for support: {round(V, 3)}")
                #st.write(data)

                add_spacelines(1)
                if pval2 < 0.05:
                    if coef2 < 0:
                        st.write('There is a negative correlation between ethotic supports and time.')
                        st.write('It seems that people tend to use **ethotic supports at the beggining of a discussion**.')
                    if coef2 > 0:
                        st.write('There is a positive correlation between ethotic supports and time.')
                        st.write('It seems that people tend to use **ethotic supports at the end of a discussion**.')
                else:
                    st.write('No statistically significant results.')

            with col2_corr:
                # attack
                con_id = []
                att = []
                for cid in df1.conversation_id.unique():
                    if 1 in df1[df1.conversation_id == cid].attack.unique():
                        df11 = df1[df1.conversation_id == cid].groupby(['time'], as_index=False)['attack'].max()
                        con_id.extend( list(df11.index) )
                        att.extend(df11.attack.to_list())
                coef22, pval22 = pointbiserialr(con_id, att)

                if pval22 >= 0.05:
                    #st.subheader("Correlation between ethotic attacks and time")
                    col2_corr.metric("Correlation between ethotic attacks and time",
                                value = round(coef22, 3), delta = round(pval22, 4), delta_color="inverse")
                elif pval22 < 0.05:
                    #st.subheader("Correlation between ethotic attacks and time")
                    col2_corr.metric("Correlation between ethotic attacks and time",
                                value = round(coef22, 3), delta = round(pval22, 4))

                part1_sup = 0
                part1_neu = 0
                part2_sup = 0
                part2_neu = 0
                for cid in df1.conversation_id.unique():
                    if 1 in df1[df1.conversation_id == cid].attack.unique():
                        partition = int(len(df1[df1.conversation_id == cid])/2)
                        part1 = df1[df1.conversation_id == cid].iloc[:partition]
                        part2 = df1[df1.conversation_id == cid].iloc[partition:]
                        part1_sup += part1.attack.sum()
                        part2_sup += part2.attack.sum()
                        part1_neu += part1.shape[0] - part1.attack.sum()
                        part2_neu += part2.shape[0] - part2.attack.sum()

                data = np.array([[part1_neu, part1_sup], [part2_neu, part2_sup]])
                #Chi-squared test statistic, sample size, and minimum of rows and columns
                X2 = stats.chi2_contingency(data, correction=False)[0]
                n = np.sum(data)
                minDim = min(data.shape)-1
                #calculate Cramer's V
                V = np.sqrt((X2/n) / minDim)
                #st.write(f"Cramer's V for attack: {round(V, 3)}")
                #st.write(data)

                add_spacelines(1)
                if pval22 < 0.05:
                    if coef22 < 0:
                        st.write('There is a negative correlation between ethotic attacks and time.')
                        st.write('It seems that people tend to use **ethotic attacks at the beggining of a discussion**.')
                    if coef22 > 0:
                        st.write('There is a positive correlation between ethotic attacks and time.')
                        st.write('It seems that people tend to use **ethotic attacks at the end of a discussion**.')
                else:
                    st.write('No statistically significant results.')
            add_spacelines(2)
