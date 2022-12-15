

# cd C:\Users\User\Downloads\LEP_test-main

# python -m streamlit run app.py

# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-talk")

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS



# functions
def user_rhetoric_v2(dataframe, source_column = 'Source', ethos_col = 'ethos_name',
                  pathos_col = 'pathos_name', logos_col = 'logos_name'):

  import warnings
  from pandas.core.common import SettingWithCopyWarning
  warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  metric_value = []
  users_list = []

  map_ethos_weight = {'attack':-1, 'neutral':0, 'support':1}
  map_pathos_weight = {'negative':-1, 'neutral':0, 'positive':1}
  map_logos_weight = {'attack':-0.5, 'neutral':0, 'support':0.5}

  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]

    ethos_pathos_logos_user = 0

    df_user_rhetoric = df_user.groupby([str(pathos_col), str(logos_col), str(ethos_col)], as_index=False).size()

    # map weights
    df_user_rhetoric[pathos_col] = df_user_rhetoric[pathos_col].map(map_pathos_weight)
    df_user_rhetoric[ethos_col] = df_user_rhetoric[ethos_col].map(map_ethos_weight)
    df_user_rhetoric[logos_col] = df_user_rhetoric[logos_col].map(map_logos_weight)

    ethos_pathos_logos_sum_ids = []

    for id in df_user_rhetoric.index:
      ethos_pathos_val = np.sum(df_user_rhetoric.loc[id, str(pathos_col):str(ethos_col)].to_numpy())
      ethos_pathos_val = ethos_pathos_val * df_user_rhetoric.loc[id, 'size']
      ethos_pathos_logos_sum_ids.append(ethos_pathos_val)

    ethos_pathos_logos_user = np.sum(ethos_pathos_logos_sum_ids)
    metric_value.append(int(ethos_pathos_logos_user))

  df = pd.DataFrame({'user': users_list, 'rhetoric_metric': metric_value})
  return df


def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(collocations=False, max_words=250, colormap=colormap, width = width, height = height,
                background_color ='black',
                min_font_size = 14, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
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
    dataframe['% lexeme'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['% lexeme'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['% lexeme'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['% lexeme'] >= int(lexeme_threshold)) & (dataframe['general #'] > 2) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with % lexeme threshold equal to {lexeme_threshold}.')
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
  figure_cloud = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud


def standardize(data):
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data0 = data.copy()
  scaled_values = scaler.fit_transform(data0)
  data0.loc[:, :] = scaled_values
  return data0


# app version
def user_stats_app(dataframe, source_column = 'Source', logos_column = 'logos_name',
               ethos_column = 'ethos_name', pathos_column = 'pathos_name'):

  sources_list = dataframe[dataframe[source_column] != 'nan'][source_column].unique()
  df = pd.DataFrame(columns = ['user', 'text_n',
                               'ethos_n', 'ethos_support_n', 'ethos_attack_n',
                               'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
                               'logos_n', 'logos_support_n', 'logos_attack_n',
                             'ethos_percent', 'ethos_support_percent', 'ethos_attack_percent',
                             'pathos_percent', 'pathos_negative_percent', 'pathos_positive_percent',
                             'logos_percent', 'logos_support_percent', 'logos_attack_percent'])
  users_list = []

  for i, u in enumerate(sources_list):
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]
    N_user = int(len(df_user))

    df_user_logos = df_user.groupby(logos_column, as_index = False)["Text"].size()
    try:
      N_ra = int(df_user_logos[df_user_logos[logos_column] == 'support']['size'].iloc[0])
      N_ra = int(N_ra)
    except:
      N_ra = 0

    df_user_ca = df_user.groupby(logos_column, as_index = False)["Text"].size()
    try:
      N_ca = int(df_user_ca[df_user_ca[logos_column] == 'attack']['size'].iloc[0])
      N_ca = int(N_ca)
    except:
      N_ca = 0

    df_user_ethos = df_user.groupby(ethos_column, as_index = False)["Text"].size()
    try:
      N_support = int(df_user_ethos[df_user_ethos[ethos_column] == 'support']['size'].iloc[0])
    except:
      N_support = 0

    try:
      N_attack = int(df_user_ethos[df_user_ethos[ethos_column] == 'attack']['size'].iloc[0])
    except:
      N_attack=0

    df_user_pathos = df_user.groupby(pathos_column, as_index = False)["Text"].size()
    try:
      N_neg = int(df_user_pathos[df_user_pathos[pathos_column] == 'negative']['size'].iloc[0])
    except:
      N_neg = 0

    try:
      N_pos = int(df_user_pathos[df_user_pathos[pathos_column] == 'positive']['size'].iloc[0])
    except:
      N_pos = 0

    counts_list = [N_support+N_attack, N_support, N_attack, N_neg+N_pos, N_neg, N_pos, N_ra+N_ca, N_ra, N_ca]
    percent_list = list((np.array(counts_list) / N_user).round(3) * 100)
    df.loc[i] = [u] + [N_user] + counts_list + percent_list
  return df




def plot_pathos_emo(data, title = 'Pathos - emotions analytics\n'):
    df_melt_emo_pathos = data[pathos_cols[4:]].fillna(0).melt(var_name='emotion', value_name = 'value')
    df_melt_emo_pathos_perc = df_melt_emo_pathos.groupby('emotion')['value'].mean().reset_index()
    df_melt_emo_pathos_perc['value'] = df_melt_emo_pathos_perc['value'].round(3) * 100

    color_map_pathos_emo = {'anger': '#BB0000','anticipation': '#D87D00','disgust': '#BB0000',
     'fear':'#BB0000','happiness': '#026F00','sadness': '#BB0000','surprise': '#D87D00','trust': '#026F00'}

    fig_emo = sns.catplot(data = df_melt_emo_pathos_perc.sort_values(by = 'value', ascending=False),
                x = 'value', y = 'emotion', hue = 'emotion', kind = 'bar',
                aspect=1.8, dodge=False, palette = color_map_pathos_emo, height = 6, legend=False)
    plt.xlabel('\npercentage %', fontsize=18)
    plt.ylabel(' ')
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, df_melt_emo_pathos_perc['value'].max()+3, 2), fontsize=15)
    plt.xlim(0, df_melt_emo_pathos_perc['value'].max()+3)
    plt.title(title, fontsize=23)
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#BB0000', label='negative emotion')
    green_patch = mpatches.Patch(color='#026F00', label='positive emotion')
    gold_patch = mpatches.Patch(color='#D87D00', label='ambivalent emotion')
    plt.legend(handles=[red_patch, green_patch, gold_patch], fontsize=15, loc = 'lower right')
    plt.show()
    return fig_emo

def plot_pathos_emo_counts(data, title = 'Pathos - emotions analytics\n'):
    df_melt_emo_pathos = data[pathos_cols[4:]].fillna(0).melt(var_name='emotion', value_name = 'value')
    df_melt_emo_pathos_perc = df_melt_emo_pathos.groupby(['emotion', 'value'], as_index=False).size()
    df_melt_emo_pathos_perc = df_melt_emo_pathos_perc[df_melt_emo_pathos_perc.value == 1]

    color_map_pathos_emo = {'anger': '#BB0000','anticipation': '#D87D00','disgust': '#BB0000',
     'fear':'#BB0000','happiness': '#026F00','sadness': '#BB0000','surprise': '#D87D00','trust': '#026F00'}

    fig_emo = sns.catplot(data = df_melt_emo_pathos_perc.sort_values(by = 'size', ascending=False),
                x = 'size', y = 'emotion', hue = 'emotion', kind = 'bar',
                aspect=1.8, dodge=False, palette = color_map_pathos_emo, height = 6, legend=False)
    plt.xlabel('\nnumber', fontsize=18)
    plt.ylabel(' ')
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, df_melt_emo_pathos_perc['size'].max()+25, 50), fontsize=15)
    plt.xlim(0, df_melt_emo_pathos_perc['size'].max()+25)
    plt.title(title, fontsize=23)
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='#BB0000', label='negative emotion')
    green_patch = mpatches.Patch(color='#026F00', label='positive emotion')
    gold_patch = mpatches.Patch(color='#D87D00', label='ambivalent emotion')
    plt.legend(handles=[red_patch, green_patch, gold_patch], fontsize=15, loc = 'lower right')
    plt.show()
    return fig_emo


def plot_rhetoric_basic_stats1_post(df, var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)

    data = df.reset_index().rename(columns={"index":"Sentence_id"})
    if not datasets_singles_hansard_logos:
        data['Attack'] = np.where(data['ethos_name'] == 'attack', 1, 0)
        data['Support'] = np.where(data['ethos_name'] == 'support', 1, 0)
    if not datasets_singles_hansard_ethos:
        data['Logos_attack'] = np.where(data['logos_name'] == 'attack', 1, 0)
        data['Logos_support'] = np.where(data['logos_name'] == 'support', 1, 0)

    if datasets_singles_hansard_ethos:
        cols_all = ["Sentence_id", 'Ethos_Label', 'Support', 'Attack']
        cols_11 = ['map_ID',"new_id",'Text', 'Source', 'clean_Text_lemmatized', 'Target']
    elif datasets_singles_hansard_logos:
        cols_all = ["Sentence_id", 'Logos_attack', 'Logos_support']
        cols_11 = ['map_ID',"new_id",'Text', 'Source', 'clean_Text_lemmatized']
    else:
        cols_all = ["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence', 'happiness', 'anger',
                    'sadness', 'fear', 'disgust', 'surprise', 'trust','anticipation', 'Contains_ethos',
                    'Support', 'Attack', 'Ethos_Label', 'Logos_attack', 'Logos_support']
        cols_11 = ['map_ID',"new_id",'Text', 'Source', 'clean_Text_lemmatized', 'Target']

    data[cols_all] = data[cols_all].fillna(0).astype(int)
    data["map_ID"] = data["map_ID"].astype(str)
    source = data.Source.unique()
    data1 = data.copy()
    data1["new_id"] = data1["Sentence_id"]
    for i in range(len(data1)-1):
        if (data1.iloc[i+1]["Sentence_id"] == data1.iloc[i]["Sentence_id"]+1) and (data1.iloc[i]["Source"] == data1.iloc[i+1]["Source"]) and (data1.iloc[i]["map_ID"] == data1.iloc[i+1]["map_ID"]):
            data1.loc[i+1,"new_id"] = data1.loc[i,"new_id"]
    data1 = data1.astype(str)
    data2 = data1.groupby(['map_ID',"Source","new_id"])[cols_11].transform(lambda x: ", ".join(x))
    data2 = data2.drop_duplicates()

    for c in ['map_ID',"new_id", 'Source']:
        data2[c] = data2[c].apply(lambda x: x.split(", ")[0])
    if not datasets_singles_hansard_logos:
        data2['Target'] = data2['Target'].apply(lambda x: set(x.split(", ")))
        data2['Target'] = data2['Target'].apply(lambda x: [y for y in x if y != 'nan'])
    for c in cols_all:
        data1[c] = data1[c].apply(int)

    data11 = data1.groupby(['map_ID', "Source", "new_id"], as_index=False)[cols_all[1:]].sum()
    data11 = data11.drop_duplicates()
    for c in cols_all[1:]:
           data11[c] = np.where(data11[c] > 0, 1, 0)

    data = data2.merge(data11, on = ["map_ID", "new_id", "Source"])
    var_name1 = var_multiselect[0]
    if 'ethos' in var_name1:
        att_column = 'Attack'
        sup_column = 'Support'
    elif 'logos' in var_name1:
        att_column = 'Logos_attack'
        sup_column = 'Logos_support'
    elif 'pathos' in var_name1:
        att_column = 'negative_valence'
        sup_column = 'positive_valence'

    if val_type == "number":
        #st.write(data.shape)
        df_prop1a = pd.DataFrame(data[att_column].value_counts())
        df_prop1a.columns = ['count']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'count']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column].value_counts())
        df_prop1s.columns = ['count']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'count']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        #df_prop1s['label'] = df_prop1s['label'].map({1:  'support'})
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column] == 0) & (data[att_column] == 0) ])
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut]})
        df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop1 = df_prop1.reset_index(drop=True)
        df_prop1 = df_prop1.sort_values(by = 'label')


        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #st.write(data.shape)
        df_prop1a = pd.DataFrame(data[att_column].value_counts(normalize=True).round(3)*100)
        df_prop1a.columns = ['percentage']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'percentage']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        #df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column].value_counts(normalize=True).round(3)*100)
        df_prop1s.columns = ['percentage']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'percentage']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        #df_prop1s['label'] = df_prop1s['label'].map({1:  'support'})
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column] == 0) & (data[att_column] == 0) ])
        n_neut = round(n_neut / len(data) * 100, 1)
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut]})
        df_prop = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop = df_prop.reset_index(drop=True)
        df_prop = df_prop.sort_values(by = 'label')

        title_str = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop['label'], df_prop['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage %\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str} analytics\n", fontsize=23)
        vals0 = df_prop['percentage'].values
        for i, v in enumerate(vals0):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column] == 1) | (data[sup_column] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column] == 1) | (data[sup_column] == 1) ]) * 100, 1)
        if 'pathos' in var_name1:
            df_prop11 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop11 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop11 = df_prop11.sort_values(by = 'label')

        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage %\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals1 = df_prop11['percentage'].values
        for i, v in enumerate(vals1):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1


def plot_rhetoric_basic_stats1(var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    if val_type == "number":
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        var_name1 = var_multiselect[0]
        df_prop = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop.columns = ['percentage']
        df_prop.reset_index(inplace=True)
        df_prop.columns = ['label', 'percentage']
        df_prop = df_prop.sort_values(by = 'label')

        title_str = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop['label'], df_prop['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str} analytics\n", fontsize=23)
        vals0 = df_prop['percentage'].values
        for i, v in enumerate(vals0):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        df_prop11 = pd.DataFrame(df[df[str(var_name1)] != 'neutral'][str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop11.columns = ['percentage']
        df_prop11.reset_index(inplace=True)
        df_prop11.columns = ['label', 'percentage']
        df_prop11 = df_prop11.sort_values(by = 'label')
        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals1 = df_prop11['percentage'].values
        for i, v in enumerate(vals1):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1



def plot_rhetoric_basic_stats(var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    fig_list = []
    for i in range(num_vars):
        var_name1 = var_multiselect[i]

        if val_type == "number":
            #plot1
            var_name1 = var_multiselect[0]
            df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
            df_prop1.columns = ['count']
            df_prop1.reset_index(inplace=True)
            df_prop1.columns = ['label', 'count']
            #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
            df_prop1 = df_prop1.sort_values(by = 'label')

            title_str1 = str(var_name1).replace('_name', '').capitalize()
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
            plt.ylabel('number\n', fontsize=16)
            plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
            plt.xticks(fontsize=17)
            plt.title(f"{title_str1} analytics\n", fontsize=23)
            vals1 = df_prop1['count'].values
            for i, v in enumerate(vals1):
                plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
            plt.show()
            fig_list.append(fig1)

        else:
            #plot1
            var_name1 = var_multiselect[0]
            df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
            df_prop1.columns = ['percentage']
            df_prop1.reset_index(inplace=True)
            df_prop1.columns = ['label', 'percentage']
            #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
            df_prop1 = df_prop1.sort_values(by = 'label')

            title_str1 = str(var_name1).replace('_name', '').capitalize()
            fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
            ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
            ax1[0].set_ylabel('percentage\n', fontsize=16)
            ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
            ax1[0].tick_params(axis='x', labelsize=17)
            ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
            vals1 = df_prop1['percentage'].values
            for i, v in enumerate(vals1):
                ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
            df_prop11 = pd.DataFrame(df[df[str(var_name1)] != 'neutral'][str(var_name1)].value_counts(normalize=True).round(3)*100)
            df_prop11.columns = ['percentage']
            df_prop11.reset_index(inplace=True)
            df_prop11.columns = ['label', 'percentage']
            df_prop11 = df_prop11.sort_values(by = 'label')
            ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
            ax1[1].set_ylabel('percentage\n', fontsize=16)
            ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
            ax1[1].tick_params(axis='x', labelsize=17)
            ax1[1].set_title(f"\n", fontsize=23)
            vals11 = df_prop11['percentage'].values
            for i, v in enumerate(vals11):
                ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
            plt.tight_layout()
            plt.show()
            fig_list.append(fig1)
        #for fig in fig_list st.pyplot(fig)
    return fig_list



def plot_rhetoric_basic_stats_post(dataframe, var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)

    data = dataframe.reset_index().rename(columns={"index":"Sentence_id"})
    data['Logos_attack'] = np.where(data['ethos_name'] == 'attack', 1, 0)
    data['Logos_support'] = np.where(data['ethos_name'] == 'support', 1, 0)

    data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']] = data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
            'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']].fillna(0).astype(int)
    data["map_ID"] = data["map_ID"].astype(str)
    source = data.Source.unique()
    data1 = data.copy()
    data1["new_id"] = data1["Sentence_id"]
    for i in range(len(data1)-1):
        if (data1.iloc[i+1]["Sentence_id"] == data1.iloc[i]["Sentence_id"]+1) and (data1.iloc[i]["Source"] == data1.iloc[i+1]["Source"]) and (data1.iloc[i]["map_ID"] == data1.iloc[i+1]["map_ID"]):
            data1.loc[i+1,"new_id"] = data1.loc[i,"new_id"]
    data1 = data1.astype(str)
    data2 = data1.groupby(['map_ID',"Source","new_id"])['map_ID',"new_id",'Text', 'Source',
                                                    'clean_Text_lemmatized', 'Target'].transform(lambda x: ", ".join(x))
    data2 = data2.drop_duplicates()

    for c in ['map_ID',"new_id", 'Source']:
        data2[c] = data2[c].apply(lambda x: x.split(", ")[0])
    data2['Target'] = data2['Target'].apply(lambda x: set(x.split(", ")))
    data2['Target'] = data2['Target'].apply(lambda x: [y for y in x if y != 'nan'])
    for c in ['Contains_pathos','positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
              'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support',
              'Attack', 'Logos_attack', 'Logos_support']:
              data1[c] = data1[c].apply(int)

    data11 = data1.groupby(['map_ID', "Source", "new_id"], as_index=False)['Contains_pathos',
                                                        'positive_valence', 'negative_valence',
                                                        'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
                                                        'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support'].sum()
    data11 = data11.drop_duplicates()
    for c in ['Contains_pathos','positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support']:
           data11[c] = np.where(data11[c] > 0, 1, 0)
    data = data2.merge(data11, on = ["map_ID", "new_id", "Source"])

    fig_list = []
    for i in range(num_vars):
        var_name1 = var_multiselect[i]
        if 'ethos' in var_name1:
            att_column1 = 'Attack'
            sup_column1 = 'Support'
        elif 'logos' in var_name1:
            att_column1 = 'Logos_attack'
            sup_column1 = 'Logos_support'
        elif 'pathos' in var_name1:
            att_column1 = 'negative_valence'
            sup_column1 = 'positive_valence'


        if val_type == "number":
            #plot1
            df_prop1a = pd.DataFrame(data[att_column1].value_counts())
            df_prop1a.columns = ['count']
            df_prop1a.reset_index(inplace=True)
            df_prop1a.columns = ['label', 'count']
            df_prop1a = df_prop1a[df_prop1a.label == 1]
            if 'pathos' in var_name1:
                df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
            else:
                df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

            df_prop1s = pd.DataFrame(data[sup_column1].value_counts())
            df_prop1s.columns = ['count']
            df_prop1s.reset_index(inplace=True)
            df_prop1s.columns = ['label', 'count']
            df_prop1s = df_prop1s[df_prop1s.label == 1]
            if 'pathos' in var_name1:
                df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
            else:
                df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

            n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
            df_prop1n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut]})
            df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
            df_prop1 = df_prop1.reset_index(drop=True)
            df_prop1 = df_prop1.sort_values(by = 'label')

            title_str1 = str(var_name1).replace('_name', '').capitalize()
            fig1, ax1 = plt.subplots(figsize=(9, 6))
            ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
            plt.ylabel('number\n', fontsize=16)
            plt.yticks(np.arange(0, df_prop1['count'].max()+206, 200), fontsize=15)
            plt.xticks(fontsize=17)
            plt.title(f"{title_str1} analytics\n", fontsize=23)
            vals1 = df_prop1['count'].values
            for i, v in enumerate(vals1):
                plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
            plt.show()
            fig_list.append(fig1)


        else:
            #plot1
            df_prop1a = pd.DataFrame(data[att_column1].value_counts(normalize=True).round(3)*100)
            df_prop1a.columns = ['percentage']
            df_prop1a.reset_index(inplace=True)
            df_prop1a.columns = ['label', 'percentage']
            df_prop1a = df_prop1a[df_prop1a.label == 1]
            if 'pathos' in var_name1:
                df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
            else:
                df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

            df_prop1s = pd.DataFrame(data[sup_column1].value_counts(normalize=True).round(3)*100)
            df_prop1s.columns = ['percentage']
            df_prop1s.reset_index(inplace=True)
            df_prop1s.columns = ['label', 'percentage']
            df_prop1s = df_prop1s[df_prop1s.label == 1]
            if 'pathos' in var_name1:
                df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
            else:
                df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

            n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
            n_neut = round(n_neut / len(data) * 100, 1)
            df_prop1n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut]})
            df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
            df_prop1 = df_prop1.reset_index(drop=True)
            df_prop1 = df_prop1.sort_values(by = 'label')

            title_str1 = str(var_name1).replace('_name', '').capitalize()
            fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
            ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
            ax1[0].set_ylabel('percentage %\n', fontsize=16)
            ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
            ax1[0].tick_params(axis='x', labelsize=17)
            ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
            vals1 = df_prop1['percentage'].values
            for i, v in enumerate(vals1):
                ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

            n_att = len(data[ (data[att_column1] == 1) ])
            n_att = round(n_att / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
            n_sup = len(data[ (data[sup_column1] == 1) ])
            n_sup = round(n_sup / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
            if 'pathos' in var_name1:
                df_prop11 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
            else:
                df_prop11 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
            df_prop11 = df_prop11.sort_values(by = 'label')

            ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
            ax1[1].set_ylabel('percentage %\n', fontsize=16)
            ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
            ax1[1].tick_params(axis='x', labelsize=17)
            ax1[1].set_title(f"\n", fontsize=23)
            vals11 = df_prop11['percentage'].values
            for i, v in enumerate(vals11):
                ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
            plt.tight_layout()
            plt.show()
            fig_list.append(fig1)
    # for fig in fig_list st.pyplot(fig)
    return fig_list




def plot_rhetoric_basic_stats2_post(df, var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    data = df.reset_index().rename(columns={"index":"Sentence_id"})
    data['Logos_attack'] = np.where(data['ethos_name'] == 'attack', 1, 0)
    data['Logos_support'] = np.where(data['ethos_name'] == 'support', 1, 0)

    data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']] = data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
            'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']].fillna(0).astype(int)
    data["map_ID"] = data["map_ID"].astype(str)
    source = data.Source.unique()
    data1 = data.copy()
    data1["new_id"] = data1["Sentence_id"]
    for i in range(len(data1)-1):
        if (data1.iloc[i+1]["Sentence_id"] == data1.iloc[i]["Sentence_id"]+1) and (data1.iloc[i]["Source"] == data1.iloc[i+1]["Source"]) and (data1.iloc[i]["map_ID"] == data1.iloc[i+1]["map_ID"]):
            data1.loc[i+1,"new_id"] = data1.loc[i,"new_id"]
    data1 = data1.astype(str)
    data2 = data1.groupby(['map_ID',"Source","new_id"])['map_ID',"new_id",'Text', 'Source',
                                                    'clean_Text_lemmatized', 'Target'].transform(lambda x: ", ".join(x))
    data2 = data2.drop_duplicates()

    for c in ['map_ID',"new_id", 'Source']:
        data2[c] = data2[c].apply(lambda x: x.split(", ")[0])
    data2['Target'] = data2['Target'].apply(lambda x: set(x.split(", ")))
    data2['Target'] = data2['Target'].apply(lambda x: [y for y in x if y != 'nan'])
    for c in ['Contains_pathos','positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
              'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support',
              'Attack', 'Logos_attack', 'Logos_support']:
              data1[c] = data1[c].apply(int)

    data11 = data1.groupby(['map_ID', "Source", "new_id"], as_index=False)['Contains_pathos',
                                                        'positive_valence', 'negative_valence',
                                                        'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
                                                        'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support'].sum()
    data11 = data11.drop_duplicates()
    for c in ['Contains_pathos','positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support']:
           data11[c] = np.where(data11[c] > 0, 1, 0)
    data = data2.merge(data11, on = ["map_ID", "new_id", "Source"])

    var_name1 = var_multiselect[0]
    var_name2 = var_multiselect[1]
    if 'ethos' in var_name1:
        att_column1 = 'Attack'
        sup_column1 = 'Support'
    elif 'logos' in var_name1:
        att_column1 = 'Logos_attack'
        sup_column1 = 'Logos_support'
    elif 'pathos' in var_name1:
        att_column1 = 'negative_valence'
        sup_column1 = 'positive_valence'

    if 'ethos' in var_name2:
        att_column2 = 'Attack'
        sup_column2 = 'Support'
    elif 'logos' in var_name2:
        att_column2 = 'Logos_attack'
        sup_column2 = 'Logos_support'
    elif 'pathos' in var_name2:
        att_column2 = 'negative_valence'
        sup_column2 = 'positive_valence'

    if val_type == "number":
        #plot1
        df_prop1a = pd.DataFrame(data[att_column1].value_counts())
        df_prop1a.columns = ['count']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'count']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column1].value_counts())
        df_prop1s.columns = ['count']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'count']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut]})
        df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop1 = df_prop1.reset_index(drop=True)
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        df_prop2a = pd.DataFrame(data[att_column2].value_counts())
        df_prop2a.columns = ['count']
        df_prop2a.reset_index(inplace=True)
        df_prop2a.columns = ['label', 'count']
        df_prop2a = df_prop2a[df_prop2a.label == 1]
        if 'pathos' in var_name2:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'negative'})
        else:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'attack'})

        df_prop2s = pd.DataFrame(data[sup_column2].value_counts())
        df_prop2s.columns = ['count']
        df_prop2s.reset_index(inplace=True)
        df_prop2s.columns = ['label', 'count']
        df_prop2s = df_prop2s[df_prop2s.label == 1]
        if 'pathos' in var_name2:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'positive'})
        else:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'support'})

        n_neut2 = len(data[ (data[sup_column2] == 0) & (data[att_column2] == 0) ])
        df_prop2n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut2]})
        df_prop2 = pd.concat([df_prop2a, df_prop2s, df_prop2n], axis = 0)
        df_prop2 = df_prop2.reset_index(drop=True)
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        df_prop1a = pd.DataFrame(data[att_column1].value_counts(normalize=True).round(3)*100)
        df_prop1a.columns = ['percentage']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'percentage']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column1].value_counts(normalize=True).round(3)*100)
        df_prop1s.columns = ['percentage']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'percentage']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
        n_neut = round(n_neut / len(data) * 100, 1)
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut]})
        df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop1 = df_prop1.reset_index(drop=True)
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage %\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values
        for i, v in enumerate(vals1):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column1] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column1] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
        if 'pathos' in var_name1:
            df_prop11 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop11 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop11 = df_prop11.sort_values(by = 'label')

        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage %\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals11 = df_prop11['percentage'].values
        for i, v in enumerate(vals11):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot2
        df_prop2a = pd.DataFrame(data[att_column2].value_counts(normalize=True).round(3)*100)
        df_prop2a.columns = ['percentage']
        df_prop2a.reset_index(inplace=True)
        df_prop2a.columns = ['label', 'percentage']
        df_prop2a = df_prop2a[df_prop2a.label == 1]
        if 'pathos' in var_name2:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'negative'})
        else:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'attack'})

        df_prop2s = pd.DataFrame(data[sup_column2].value_counts(normalize=True).round(3)*100)
        df_prop2s.columns = ['percentage']
        df_prop2s.reset_index(inplace=True)
        df_prop2s.columns = ['label', 'percentage']
        df_prop2s = df_prop2s[df_prop2s.label == 1]
        if 'pathos' in var_name2:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'positive'})
        else:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'support'})

        n_neut2 = len(data[ (data[sup_column2] == 0) & (data[att_column2] == 0) ])
        n_neut2 = round(n_neut2 / len(data) * 100, 1)
        df_prop2n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut2]})
        df_prop2 = pd.concat([df_prop2a, df_prop2s, df_prop2n], axis = 0)
        df_prop2 = df_prop2.reset_index(drop=True)
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))
        ax2[0].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax2[0].set_ylabel('percentage %\n', fontsize=16)
        ax2[0].set_yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        ax2[0].tick_params(axis='x', labelsize=17)
        ax2[0].set_title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values
        for i, v in enumerate(vals2):
            ax2[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column2] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column2] == 1) | (data[sup_column2] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column2] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column2] == 1) | (data[sup_column2] == 1) ]) * 100, 1)
        if 'pathos' in var_name2:
            df_prop21 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop21 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop21 = df_prop21.sort_values(by = 'label')

        ax2[1].bar(df_prop21['label'], df_prop21['percentage'], color = ['#BB0000', '#026F00'])
        ax2[1].set_ylabel('percentage %\n', fontsize=16)
        ax2[1].set_yticks(np.arange(0, df_prop21['percentage'].max()+11, 10), fontsize=15)
        ax2[1].tick_params(axis='x', labelsize=17)
        ax2[1].set_title(f"\n", fontsize=23)
        vals22 = df_prop21['percentage'].values
        for i, v in enumerate(vals22):
            ax2[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1, fig2


def plot_rhetoric_basic_stats2(var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    if val_type == "number":
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts())
        df_prop2.columns = ['count']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'count']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values
        for i, v in enumerate(vals1):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
        df_prop11 = pd.DataFrame(df[df[str(var_name1)] != 'neutral'][str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop11.columns = ['percentage']
        df_prop11.reset_index(inplace=True)
        df_prop11.columns = ['label', 'percentage']
        df_prop11 = df_prop11.sort_values(by = 'label')
        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals11 = df_prop11['percentage'].values
        for i, v in enumerate(vals11):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))
        ax2[0].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax2[0].set_ylabel('percentage\n', fontsize=16)
        ax2[0].set_yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        ax2[0].tick_params(axis='x', labelsize=17)
        ax2[0].set_title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values
        for i, v in enumerate(vals2):
            ax2[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
        df_prop22 = pd.DataFrame(df[df[str(var_name2)] != 'neutral'][str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop22.columns = ['percentage']
        df_prop22.reset_index(inplace=True)
        df_prop22.columns = ['label', 'percentage']
        df_prop22 = df_prop22.sort_values(by = 'label')
        ax2[1].bar(df_prop22['label'], df_prop22['percentage'], color = ['#BB0000', '#026F00'])
        ax2[1].set_ylabel('percentage\n', fontsize=16)
        ax2[1].set_yticks(np.arange(0, df_prop22['percentage'].max()+11, 10), fontsize=15)
        ax2[1].tick_params(axis='x', labelsize=17)
        ax2[1].set_title(f"\n", fontsize=23)
        vals22 = df_prop22['percentage'].values
        for i, v in enumerate(vals22):
            ax2[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1, fig2


def plot_rhetoric_basic_stats3(var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    if val_type == "number":
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts())
        df_prop1.columns = ['count']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'count']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts())
        df_prop2.columns = ['count']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'count']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot3
        var_name3 = var_multiselect[2]
        df_prop3 = pd.DataFrame(df[str(var_name3)].value_counts())
        df_prop3.columns = ['count']
        df_prop3.reset_index(inplace=True)
        df_prop3.columns = ['label', 'count']
        #df_prop3['label'] = df_prop3['label'].str.replace('negative', ' negative')
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.bar(df_prop3['label'], df_prop3['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop3['count'].max()+206, 500), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['count'].values
        for i, v in enumerate(vals3):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        var_name1 = var_multiselect[0]
        df_prop1 = pd.DataFrame(df[str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop1.columns = ['percentage']
        df_prop1.reset_index(inplace=True)
        df_prop1.columns = ['label', 'percentage']
        #df_prop1['label'] = df_prop1['label'].str.replace('negative', ' negative')
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values
        for i, v in enumerate(vals1):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
        df_prop11 = pd.DataFrame(df[df[str(var_name1)] != 'neutral'][str(var_name1)].value_counts(normalize=True).round(3)*100)
        df_prop11.columns = ['percentage']
        df_prop11.reset_index(inplace=True)
        df_prop11.columns = ['label', 'percentage']
        df_prop11 = df_prop11.sort_values(by = 'label')
        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals11 = df_prop11['percentage'].values
        for i, v in enumerate(vals11):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot2
        var_name2 = var_multiselect[1]
        df_prop2 = pd.DataFrame(df[str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop2.columns = ['percentage']
        df_prop2.reset_index(inplace=True)
        df_prop2.columns = ['label', 'percentage']
        #df_prop2['label'] = df_prop2['label'].str.replace('negative', ' negative')
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))
        ax2[0].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax2[0].set_ylabel('percentage\n', fontsize=16)
        ax2[0].set_yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        ax2[0].tick_params(axis='x', labelsize=17)
        ax2[0].set_title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values
        for i, v in enumerate(vals2):
            ax2[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
        df_prop22 = pd.DataFrame(df[df[str(var_name2)] != 'neutral'][str(var_name2)].value_counts(normalize=True).round(3)*100)
        df_prop22.columns = ['percentage']
        df_prop22.reset_index(inplace=True)
        df_prop22.columns = ['label', 'percentage']
        df_prop22 = df_prop22.sort_values(by = 'label')
        ax2[1].bar(df_prop22['label'], df_prop22['percentage'], color = ['#BB0000', '#026F00'])
        ax2[1].set_ylabel('percentage\n', fontsize=16)
        ax2[1].set_yticks(np.arange(0, df_prop22['percentage'].max()+11, 10), fontsize=15)
        ax2[1].tick_params(axis='x', labelsize=17)
        ax2[1].set_title(f"\n", fontsize=23)
        vals22 = df_prop22['percentage'].values
        for i, v in enumerate(vals22):
            ax2[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot3
        var_name3 = var_multiselect[2]
        df_prop3 = pd.DataFrame(df[str(var_name3)].value_counts(normalize=True).round(3)*100)
        df_prop3.columns = ['percentage']
        df_prop3.reset_index(inplace=True)
        df_prop3.columns = ['label', 'percentage']
        #df_prop3['label'] = df_prop3['label'].str.replace('negative', ' negative')
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 12))
        ax3[0].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax3[0].set_ylabel('percentage\n', fontsize=16)
        ax3[0].set_yticks(np.arange(0, df_prop3['percentage'].max()+16, 10), fontsize=15)
        ax3[0].tick_params(axis='x', labelsize=17)
        ax3[0].set_title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['percentage'].values
        for i, v in enumerate(vals3):
            ax3[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))
        df_prop33 = pd.DataFrame(df[df[str(var_name3)] != 'neutral'][str(var_name3)].value_counts(normalize=True).round(3)*100)
        df_prop33.columns = ['percentage']
        df_prop33.reset_index(inplace=True)
        df_prop33.columns = ['label', 'percentage']
        df_prop33 = df_prop33.sort_values(by = 'label')
        ax3[1].bar(df_prop33['label'], df_prop33['percentage'], color = ['#BB0000', '#026F00'])
        ax3[1].set_ylabel('percentage\n', fontsize=16)
        ax3[1].set_yticks(np.arange(0, df_prop33['percentage'].max()+11, 10), fontsize=15)
        ax3[1].tick_params(axis='x', labelsize=17)
        ax3[1].set_title(f"\n", fontsize=23)
        vals33 = df_prop33['percentage'].values
        for i, v in enumerate(vals33):
            ax3[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1, fig2, fig3

def plot_rhetoric_basic_stats3_post(df, var_multiselect, val_type = "percentage"):
    num_vars = len(var_multiselect)
    data = df.reset_index().rename(columns={"index":"Sentence_id"})
    data['Logos_attack'] = np.where(data['ethos_name'] == 'attack', 1, 0)
    data['Logos_support'] = np.where(data['ethos_name'] == 'support', 1, 0)

    data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']] = data[["Sentence_id", 'Contains_pathos', 'positive_valence', 'negative_valence',
            'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Ethos_Label']].fillna(0).astype(int)
    data["map_ID"] = data["map_ID"].astype(str)
    source = data.Source.unique()
    data1 = data.copy()
    data1["new_id"] = data1["Sentence_id"]
    for i in range(len(data1)-1):
        if (data1.iloc[i+1]["Sentence_id"] == data1.iloc[i]["Sentence_id"]+1) and (data1.iloc[i]["Source"] == data1.iloc[i+1]["Source"]) and (data1.iloc[i]["map_ID"] == data1.iloc[i+1]["map_ID"]):
            data1.loc[i+1,"new_id"] = data1.loc[i,"new_id"]
    data1 = data1.astype(str)
    data2 = data1.groupby(['map_ID',"Source","new_id"])['map_ID',"new_id",'Text', 'Source',
                                                    'clean_Text_lemmatized', 'Target'].transform(lambda x: ", ".join(x))
    data2 = data2.drop_duplicates()

    for c in ['map_ID',"new_id", 'Source']:
        data2[c] = data2[c].apply(lambda x: x.split(", ")[0])
    data2['Target'] = data2['Target'].apply(lambda x: set(x.split(", ")))
    data2['Target'] = data2['Target'].apply(lambda x: [y for y in x if y != 'nan'])
    for c in ['Contains_pathos','positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
              'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'Contains_ethos', 'Support',
              'Attack', 'Logos_attack', 'Logos_support']:
              data1[c] = data1[c].apply(int)

    data11 = data1.groupby(['map_ID', "Source", "new_id"], as_index=False)['Contains_pathos',
                                                        'positive_valence', 'negative_valence',
                                                        'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
                                                        'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support'].sum()
    data11 = data11.drop_duplicates()
    for c in ['Contains_pathos','positive_valence', 'negative_valence',
           'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust',
           'anticipation', 'Contains_ethos', 'Support', 'Attack', 'Logos_attack', 'Logos_support']:
           data11[c] = np.where(data11[c] > 0, 1, 0)
    data = data2.merge(data11, on = ["map_ID", "new_id", "Source"])

    var_name1 = var_multiselect[0]
    var_name2 = var_multiselect[1]
    var_name3 = var_multiselect[2]
    if 'ethos' in var_name1:
        att_column1 = 'Attack'
        sup_column1 = 'Support'
    elif 'logos' in var_name1:
        att_column1 = 'Logos_attack'
        sup_column1 = 'Logos_support'
    elif 'pathos' in var_name1:
        att_column1 = 'negative_valence'
        sup_column1 = 'positive_valence'

    if 'ethos' in var_name2:
        att_column2 = 'Attack'
        sup_column2 = 'Support'
    elif 'logos' in var_name2:
        att_column2 = 'Logos_attack'
        sup_column2 = 'Logos_support'
    elif 'pathos' in var_name2:
        att_column2 = 'negative_valence'
        sup_column2 = 'positive_valence'

    if 'ethos' in var_name3:
        att_column3 = 'Attack'
        sup_column3 = 'Support'
    elif 'logos' in var_name3:
        att_column3 = 'Logos_attack'
        sup_column3 = 'Logos_support'
    elif 'pathos' in var_name3:
        att_column3 = 'negative_valence'
        sup_column3 = 'positive_valence'

    if val_type == "number":
        #plot1
        df_prop1a = pd.DataFrame(data[att_column1].value_counts())
        df_prop1a.columns = ['count']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'count']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column1].value_counts())
        df_prop1s.columns = ['count']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'count']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut]})
        df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop1 = df_prop1.reset_index(drop=True)
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.bar(df_prop1['label'], df_prop1['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop1['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['count'].values
        for i, v in enumerate(vals1):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot2
        df_prop2a = pd.DataFrame(data[att_column2].value_counts())
        df_prop2a.columns = ['count']
        df_prop2a.reset_index(inplace=True)
        df_prop2a.columns = ['label', 'count']
        df_prop2a = df_prop2a[df_prop2a.label == 1]
        if 'pathos' in var_name2:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'negative'})
        else:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'attack'})

        df_prop2s = pd.DataFrame(data[sup_column2].value_counts())
        df_prop2s.columns = ['count']
        df_prop2s.reset_index(inplace=True)
        df_prop2s.columns = ['label', 'count']
        df_prop2s = df_prop2s[df_prop2s.label == 1]
        if 'pathos' in var_name2:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'positive'})
        else:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'support'})

        n_neut2 = len(data[ (data[sup_column2] == 0) & (data[att_column2] == 0) ])
        df_prop2n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut2]})
        df_prop2 = pd.concat([df_prop2a, df_prop2s, df_prop2n], axis = 0)
        df_prop2 = df_prop2.reset_index(drop=True)
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.bar(df_prop2['label'], df_prop2['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop2['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['count'].values
        for i, v in enumerate(vals2):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

        #plot3
        df_prop3a = pd.DataFrame(data[att_column3].value_counts())
        df_prop3a.columns = ['count']
        df_prop3a.reset_index(inplace=True)
        df_prop3a.columns = ['label', 'count']
        df_prop3a = df_prop3a[df_prop3a.label == 1]
        if 'pathos' in var_name3:
            df_prop3a['label'] = df_prop3a['label'].map({1: 'negative'})
        else:
            df_prop3a['label'] = df_prop3a['label'].map({1: 'attack'})

        df_prop3s = pd.DataFrame(data[sup_column3].value_counts())
        df_prop3s.columns = ['count']
        df_prop3s.reset_index(inplace=True)
        df_prop3s.columns = ['label', 'count']
        df_prop3s = df_prop3s[df_prop3s.label == 1]
        if 'pathos' in var_name3:
            df_prop3s['label'] = df_prop3s['label'].map({1: 'positive'})
        else:
            df_prop3s['label'] = df_prop3s['label'].map({1: 'support'})

        n_neut3 = len(data[ (data[sup_column3] == 0) & (data[att_column3] == 0) ])
        df_prop3n = pd.DataFrame({'label': ['neutral'], 'count': [n_neut3]})
        df_prop3 = pd.concat([df_prop3a, df_prop3s, df_prop3n], axis = 0)
        df_prop3 = df_prop3.reset_index(drop=True)
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.bar(df_prop3['label'], df_prop3['count'], color = ['#BB0000', '#022D96', '#026F00'])
        plt.ylabel('number\n', fontsize=16)
        plt.yticks(np.arange(0, df_prop3['count'].max()+206, 200), fontsize=15)
        plt.xticks(fontsize=17)
        plt.title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['count'].values
        for i, v in enumerate(vals3):
            plt.text(x=i , y = v+1 , s=f"{v}" , fontdict=dict(fontsize=17, ha='center'))
        plt.show()

    else:
        #plot1
        df_prop1a = pd.DataFrame(data[att_column1].value_counts(normalize=True).round(3)*100)
        df_prop1a.columns = ['percentage']
        df_prop1a.reset_index(inplace=True)
        df_prop1a.columns = ['label', 'percentage']
        df_prop1a = df_prop1a[df_prop1a.label == 1]
        if 'pathos' in var_name1:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'negative'})
        else:
            df_prop1a['label'] = df_prop1a['label'].map({1: 'attack'})

        df_prop1s = pd.DataFrame(data[sup_column1].value_counts(normalize=True).round(3)*100)
        df_prop1s.columns = ['percentage']
        df_prop1s.reset_index(inplace=True)
        df_prop1s.columns = ['label', 'percentage']
        df_prop1s = df_prop1s[df_prop1s.label == 1]
        if 'pathos' in var_name1:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'positive'})
        else:
            df_prop1s['label'] = df_prop1s['label'].map({1: 'support'})

        n_neut = len(data[ (data[sup_column1] == 0) & (data[att_column1] == 0) ])
        n_neut = round(n_neut / len(data) * 100, 1)
        df_prop1n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut]})
        df_prop1 = pd.concat([df_prop1a, df_prop1s, df_prop1n], axis = 0)
        df_prop1 = df_prop1.reset_index(drop=True)
        df_prop1 = df_prop1.sort_values(by = 'label')

        title_str1 = str(var_name1).replace('_name', '').capitalize()
        fig1, ax1 = plt.subplots(2, 1, figsize=(10, 12))
        ax1[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax1[0].set_ylabel('percentage %\n', fontsize=16)
        ax1[0].set_yticks(np.arange(0, df_prop1['percentage'].max()+16, 10), fontsize=15)
        ax1[0].tick_params(axis='x', labelsize=17)
        ax1[0].set_title(f"{title_str1} analytics\n", fontsize=23)
        vals1 = df_prop1['percentage'].values
        for i, v in enumerate(vals1):
            ax1[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column1] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column1] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column1] == 1) | (data[sup_column1] == 1) ]) * 100, 1)
        if 'pathos' in var_name1:
            df_prop11 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop11 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop11 = df_prop11.sort_values(by = 'label')

        ax1[1].bar(df_prop11['label'], df_prop11['percentage'], color = ['#BB0000', '#026F00'])
        ax1[1].set_ylabel('percentage %\n', fontsize=16)
        ax1[1].set_yticks(np.arange(0, df_prop11['percentage'].max()+11, 10), fontsize=15)
        ax1[1].tick_params(axis='x', labelsize=17)
        ax1[1].set_title(f"\n", fontsize=23)
        vals11 = df_prop11['percentage'].values
        for i, v in enumerate(vals11):
            ax1[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot2
        df_prop2a = pd.DataFrame(data[att_column2].value_counts(normalize=True).round(3)*100)
        df_prop2a.columns = ['percentage']
        df_prop2a.reset_index(inplace=True)
        df_prop2a.columns = ['label', 'percentage']
        df_prop2a = df_prop2a[df_prop2a.label == 1]
        if 'pathos' in var_name2:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'negative'})
        else:
            df_prop2a['label'] = df_prop2a['label'].map({1: 'attack'})

        df_prop2s = pd.DataFrame(data[sup_column2].value_counts(normalize=True).round(3)*100)
        df_prop2s.columns = ['percentage']
        df_prop2s.reset_index(inplace=True)
        df_prop2s.columns = ['label', 'percentage']
        df_prop2s = df_prop2s[df_prop2s.label == 1]
        if 'pathos' in var_name2:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'positive'})
        else:
            df_prop2s['label'] = df_prop2s['label'].map({1: 'support'})

        n_neut2 = len(data[ (data[sup_column2] == 0) & (data[att_column2] == 0) ])
        n_neut2 = round(n_neut2 / len(data) * 100, 1)
        df_prop2n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut2]})
        df_prop2 = pd.concat([df_prop2a, df_prop2s, df_prop2n], axis = 0)
        df_prop2 = df_prop2.reset_index(drop=True)
        df_prop2 = df_prop2.sort_values(by = 'label')

        title_str2 = str(var_name2).replace('_name', '').capitalize()
        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 12))
        ax2[0].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax2[0].set_ylabel('percentage %\n', fontsize=16)
        ax2[0].set_yticks(np.arange(0, df_prop2['percentage'].max()+16, 10), fontsize=15)
        ax2[0].tick_params(axis='x', labelsize=17)
        ax2[0].set_title(f"{title_str2} analytics\n", fontsize=23)
        vals2 = df_prop2['percentage'].values
        for i, v in enumerate(vals2):
            ax2[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column2] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column2] == 1) | (data[sup_column2] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column2] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column2] == 1) | (data[sup_column2] == 1) ]) * 100, 1)
        if 'pathos' in var_name2:
            df_prop21 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop21 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop21 = df_prop21.sort_values(by = 'label')

        ax2[1].bar(df_prop21['label'], df_prop21['percentage'], color = ['#BB0000', '#026F00'])
        ax2[1].set_ylabel('percentage %\n', fontsize=16)
        ax2[1].set_yticks(np.arange(0, df_prop21['percentage'].max()+11, 10), fontsize=15)
        ax2[1].tick_params(axis='x', labelsize=17)
        ax2[1].set_title(f"\n", fontsize=23)
        vals22 = df_prop21['percentage'].values
        for i, v in enumerate(vals22):
            ax2[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()

        #plot3
        df_prop3a = pd.DataFrame(data[att_column3].value_counts(normalize=True).round(3)*100)
        df_prop3a.columns = ['percentage']
        df_prop3a.reset_index(inplace=True)
        df_prop3a.columns = ['label', 'percentage']
        df_prop3a = df_prop3a[df_prop3a.label == 1]
        if 'pathos' in var_name3:
            df_prop3a['label'] = df_prop3a['label'].map({1: 'negative'})
        else:
            df_prop3a['label'] = df_prop3a['label'].map({1: 'attack'})

        df_prop3s = pd.DataFrame(data[sup_column3].value_counts(normalize=True).round(3)*100)
        df_prop3s.columns = ['percentage']
        df_prop3s.reset_index(inplace=True)
        df_prop3s.columns = ['label', 'percentage']
        df_prop3s = df_prop3s[df_prop3s.label == 1]
        if 'pathos' in var_name3:
            df_prop3s['label'] = df_prop3s['label'].map({1: 'positive'})
        else:
            df_prop3s['label'] = df_prop3s['label'].map({1: 'support'})

        n_neut3 = len(data[ (data[sup_column3] == 0) & (data[att_column3] == 0) ])
        n_neut3 = round(n_neut3 / len(data) * 100, 1)
        df_prop3n = pd.DataFrame({'label': ['neutral'], 'percentage': [n_neut3]})
        df_prop3 = pd.concat([df_prop3a, df_prop3s, df_prop3n], axis = 0)
        df_prop3 = df_prop3.reset_index(drop=True)
        df_prop3 = df_prop3.sort_values(by = 'label')

        title_str3 = str(var_name3).replace('_name', '').capitalize()
        fig3, ax3  = plt.subplots(2, 1, figsize=(10, 12))
        ax3[0].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
        ax3[0].set_ylabel('percentage %\n', fontsize=16)
        ax3[0].set_yticks(np.arange(0, df_prop3['percentage'].max()+16, 10), fontsize=15)
        ax3[0].tick_params(axis='x', labelsize=17)
        ax3[0].set_title(f"{title_str3} analytics\n", fontsize=23)
        vals3 = df_prop3['percentage'].values
        for i, v in enumerate(vals3):
            ax3[0].text(x=i , y = v+1 , s=f"{round(v, 1)}%" , fontdict=dict(fontsize=17, ha='center'))

        n_att = len(data[ (data[att_column3] == 1) ])
        n_att = round(n_att / len(data[ (data[att_column3] == 1) | (data[sup_column3] == 1) ]) * 100, 1)
        n_sup = len(data[ (data[sup_column3] == 1) ])
        n_sup = round(n_sup / len(data[ (data[att_column3] == 1) | (data[sup_column3] == 1) ]) * 100, 1)
        if 'pathos' in var_name3:
            df_prop31 = pd.DataFrame({'label': ['negative', 'positive'], 'percentage': [n_att, n_sup]})
        else:
            df_prop31 = pd.DataFrame({'label': ['attack', 'support'], 'percentage': [n_att, n_sup]})
        df_prop31 = df_prop31.sort_values(by = 'label')

        ax3[1].bar(df_prop31['label'], df_prop31['percentage'], color = ['#BB0000', '#026F00'])
        ax3[1].set_ylabel('percentage %\n', fontsize=16)
        ax3[1].set_yticks(np.arange(0, df_prop31['percentage'].max()+11, 10), fontsize=15)
        ax3[1].tick_params(axis='x', labelsize=17)
        ax3[1].set_title(f"\n", fontsize=23)
        vals33 = df_prop31['percentage'].values
        for i, v in enumerate(vals33):
            ax3[1].text(x=i , y = v+1 , s=f"{round(v, 0)}%" , fontdict=dict(fontsize=17, ha='center'))
        plt.tight_layout()
        plt.show()
    return fig1, fig2, fig3


def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")


@st.cache(allow_output_mutation=True)
def load_dataset(dataset):
    if dataset == "US-Presidential-2016_Reddit":
        data = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_US2016.xlsx", index_col = 0)
    elif dataset ==  "Conspiracy-Theories-Vaccines-2021_Reddit":
        data = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_conspiracy.xlsx", index_col = 0)
    return data



pathos_cols = ['No_pathos', 'Contains_pathos',
       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
       'fear', 'disgust', 'surprise', 'trust', 'anticipation']

rhetoric_dims = ['logos', 'ethos', 'pathos']


# page config
st.set_page_config(page_title="LEP Analytics", layout="wide") # centered wide



def style_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# multi pages functions

def MainPage():
    st.title("LEPAn: Logos - Ethos - Pathos Analytics")
    add_spacelines(2)

    st.write("#### Trust Analytics in Digital Rhetoric")
    with st.expander("Read abstract"):
        #add_spacelines(1)
        st.write("""
        Trust plays a critical role in establishing intellectual humility and interpersonal civility in
    argumentation and discourse: without it, credibility is doomed, reputation is endangered,
    cooperation is compromised. The major threats associated with digitalisation – hate speech and
    fake news – are violations of the basic condition for trusting and being trustworthy which are key
    for constructive, reasonable and responsible communication as well as for the collaborative and
    ethical organisation of societies. This calls for a reliable and rich model which allows us to
    recognise, both manually and automatically, how trust is established, supported, attacked and
    destroyed in natural argumentation.

    The aim of this paper is to recognise references to (dis)trust using Artificial Intelligence with a
    linguistics, computational and analytics perspective to understand the specific language that is
    used in politics and conspiracy theories, when describing the trusted and distrusted entities, such
    as politicians and organisations. Building upon the previous work in argument analytics (Lawrence
    et al 206; 2017) and theoretical and computational language models for ethos mining (Budzynska
    and Duthie 2018; Pereira-Farina at al. 2022), the paper will create language resources and an
    annotation scheme which will allow the curation of a large corpus of references to trust in Reddit -
    specifically the subreddit dedicated to the US 2016 presidential debates and to conspiracy
    theories. Natural Language Processing techniques will be utilised to produce a computational
    model of references to trust with the ability to precisely classify unseen text as containing trust,
    distrust or neither.

    This will allow us to infer from structured data statistical patterns such as: frequencies of using
    appeals to trust expressing hate to specific persons, e.g., to Trump or Clinton; frequencies of using
    different authorities such as (pseudo-)science or law to increase the “credibility” of fake news; and,
    investigate how user’s opinions of these entities swing from positive to negative over time. These
    insights will reveal which trends are common in social media. The long-term ambition of this work
    is to contribute to the recently announced priority of the EC of Europe fit for the Digital Age.
    """)



    with st.container():
        add_spacelines(3)

        st.write("Paper related to the project: ")
        st.write("**Budzynska, K et al. (2022). Trust Analytics in Digital Rhetoric. *4th European Conference on Argumentation*.**")

        st.write("**[The New Ethos Lab](https://newethos.org/)**")
        st.write(" ************************** ")
        #hide_footer_style = """
        #    <style>
        #    footer {visibility: visible;
        #            color : white;
        #            background-color: #d2cdcd;}

        #    footer:after{
        #    visibility: visible;
        #    content : 'Project developed by: Ewelina Gajewska,  Marie-Amelie Paquin,   He Zhang';
        #    display : block;
        #    positive : relative;
        #    color : white;
        #    background-color: #d2cdcd;
        #    padding: 5px;
        #    top: 3px;
        #    font-weight: bold;
        #    }
        #    </style>
        #"""
        #st.markdown(hide_footer_style, unsafe_allow_html=True)


    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)

pathos_cols = ['No_pathos', 'Contains_pathos',
       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
       'fear', 'disgust', 'surprise', 'trust', 'anticipation']



def basicLEPAn():
    st.subheader(f" Text-Based Analysis ")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    if datasets_singles_hansard_ethos:
        var_to_plot_raw = st.multiselect("Choose rhetoric categories you would like to visualise", ['ethos'], 'ethos')
    elif datasets_singles_hansard_logos:
        var_to_plot_raw = st.multiselect("Choose rhetoric categories you would like to visualise", ['logos'], 'logos')
    else:
        var_to_plot_raw = st.multiselect("Choose rhetoric categories you would like to visualise", rhetoric_dims, rhetoric_dims[1])
    var_to_plot = [str(x).replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name") for x in var_to_plot_raw]

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    #check_rhet_dim = st.radio("Choose the unit of y-axis", ("percentage", "number"))
    if len(var_to_plot) > 0:
        cc0, col_radio1, col_radio2 = st.columns([1, 2, 2], gap="medium")
        with cc0:
            st.write("")
        with col_radio1:
            check_rhet_dim = st.radio("Choose the unit of y-axis", ("percentage", "number"))
        with col_radio2:
            if datasets_singles_hansard_ethos:
                check_rhet_dim_unit = st.radio("The unit of text", ["post"])
            elif datasets_singles_hansard_logos:
                check_rhet_dim_unit = st.radio("The unit of text", ["ADU"])
            else:
                check_rhet_dim_unit = st.radio("Choose the unit of text", ("ADU", "post"))
    else:
        check_rhet_dim = st.radio("Choose the unit of y-axis", ("percentage", "number"))
        check_rhet_dim_unit = 'ADU'

    #with st.form("my_form"):

            # Every form must have a submit button.
        #    submitted = st.form_submit_button("Show results")
    add_spacelines()

    if len(var_to_plot) == 1:
        if check_rhet_dim == "number":
            if check_rhet_dim_unit == 'post':
                fig1 = plot_rhetoric_basic_stats1_post(df, var_to_plot, val_type = "number")
            else:
                fig1 = plot_rhetoric_basic_stats1(var_to_plot, val_type = "number")
        else:
            if check_rhet_dim_unit == 'post':
                fig1 = plot_rhetoric_basic_stats1_post(df, var_to_plot)
            else:
                fig1 = plot_rhetoric_basic_stats1(var_to_plot)

        st.pyplot(fig1)
        if "pathos_name" in var_to_plot and check_rhet_dim_unit != 'post':
            add_spacelines(1)
            if check_rhet_dim == "percentage":
                fig_emo_pat = plot_pathos_emo(df)
                st.pyplot(fig_emo_pat)
            if check_rhet_dim == "number":
                fig_emo_pat = plot_pathos_emo_counts(df)
                st.pyplot(fig_emo_pat)


    elif len(var_to_plot) == 2:
        tab1, tab2 = st.tabs([x.upper() for x in var_to_plot_raw])
        if check_rhet_dim == "number":
            if check_rhet_dim_unit == 'post':
                fig1, fig2 = plot_rhetoric_basic_stats2_post(df, var_to_plot, val_type = "number")
            else:
                fig1, fig2 = plot_rhetoric_basic_stats2(var_to_plot, val_type = "number")
        else:
            if check_rhet_dim_unit == 'post':
                fig1, fig2 = plot_rhetoric_basic_stats2_post(df, var_to_plot)
            else:
                fig1, fig2 = plot_rhetoric_basic_stats2(var_to_plot)

        if "pathos_name" in var_to_plot and check_rhet_dim_unit != 'post':
            tab_id = list(var_to_plot_raw).index('pathos')
            #add_spacelines(1)
            if check_rhet_dim == "percentage":
                fig_emo_pat = plot_pathos_emo(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
            elif check_rhet_dim == "number":
                fig_emo_pat = plot_pathos_emo_counts(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
        with tab1:
            st.pyplot(fig1)
        with tab2:
            st.pyplot(fig2)

        if "pathos_name" in var_to_plot and check_rhet_dim_unit != 'post':
            if tab_id == 0:
                with tab1:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 1:
                with tab2:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)

    elif len(var_to_plot) == 3:
        tab1, tab2, tab3 = st.tabs([x.upper() for x in var_to_plot_raw])
        if check_rhet_dim == "number":
            if check_rhet_dim_unit == 'post':
                fig1, fig2, fig3 = plot_rhetoric_basic_stats3_post(df, var_to_plot, val_type = "number")
            else:
                fig1, fig2, fig3 = plot_rhetoric_basic_stats3(var_to_plot, val_type = "number")
        else:
            if check_rhet_dim_unit == 'post':
                fig1, fig2, fig3 = plot_rhetoric_basic_stats3_post(df, var_to_plot)
            else:
                fig1, fig2, fig3 = plot_rhetoric_basic_stats3(var_to_plot)

        if "pathos_name" in var_to_plot and check_rhet_dim_unit != 'post':
            tab_id = list(var_to_plot_raw).index('pathos')
            if check_rhet_dim == "percentage":
                fig_emo_pat = plot_pathos_emo(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
            elif check_rhet_dim == "number":
                fig_emo_pat = plot_pathos_emo_counts(df)
                #st.pyplot(fig_emo_pat)
                #add_spacelines(1)
        with tab1:
            st.pyplot(fig1)
        with tab2:
            st.pyplot(fig2)
        with tab3:
            st.pyplot(fig3)

        if "pathos_name" in var_to_plot and check_rhet_dim_unit != 'post':
            if tab_id == 0:
                with tab1:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 1:
                with tab2:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
            elif tab_id == 2:
                with tab3:
                    add_spacelines(2)
                    st.pyplot(fig_emo_pat)
    else:
        st.write("")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def generateWordCloud():
    #st.header(f" Text-Level Analytics ")
    st.subheader("WordCloud")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    if datasets_singles_hansard_ethos:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", ['ethos'], index=0)
    elif datasets_singles_hansard_logos:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", ['logos'], index=0)
    else:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims, index=0)
    add_spacelines(1)
    label_cloud = st.radio(
         "Choose a subcategory for a WordCloud",
         ('attack / negative', 'support / positive', 'both'))

    selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")
    label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for a WordCloud', 0, 100, 90)
    st.info(f'Selected precision: **{threshold_cloud}**')

    add_spacelines(1)

    if (selected_rhet_dim == 'ethos_name') or (selected_rhet_dim == 'logos_name'):
         df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
         df[df[str(selected_rhet_dim)] == 'support'],
         df[df[str(selected_rhet_dim)] == 'attack'])
    else:
        df_for_wordcloud = prepare_cloud_lexeme_data(df[df[str(selected_rhet_dim)] == 'neutral'],
        df[df[str(selected_rhet_dim)] == 'positive'],
        df[df[str(selected_rhet_dim)] == 'negative'])

    add_spacelines(1)
    try:
        fig_cloud = wordcloud_lexeme(df_for_wordcloud, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))
        st.pyplot(fig_cloud)
    except ValueError:
        add_spacelines(1)
        st.warning("**You need to decrease the precision value** (no words with the selected threshold).")


    add_spacelines(4)
    with st.expander("High Precision Words"):
        add_spacelines(1)
        st.write("How accurate we are with finding a text belonging to the chosen category when a particular word is present in the text.")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def TargetHeroScores():
    st.subheader(f"(Anti)Heroes - Target Entity Analysis ")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    if datasets_singles_us2016 and datasets_singles_conspiracy and not datasets_singles_hansard_ethos:
        list_targets = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
                           'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
                           'NATO', 'Republicans', 'Government','O Malley',
                            'BLM', 'Chafee', 'CNN', 'Chafee', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
                           'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin', 'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly',
                           'Carson', 'Facebook', 'Walker', 'Left', 'Conservatives', 'Trudeau',
                           'Bill Clinton', 'Holt', 'McCain', 'Supporters Trump', 'Biden', 'GOP', 'Kaplan', 'Media', 'Occupy', "Rosie O'Donnell",
                           'Americans','BBC','CNN','Dead mother','Fauci', "HolidayOk4857's husband",'JamesHollywoodSEA’s wife','MSM','New York','Not-high school graduates',
                           'People for mandatory vaccination','People taking joy in someone’s death','Pro-vaccinators','Reddit users','Reuters','Russians','Talkradio',
                           'The experts',"TheUnwillingOne's family",'Trump', 'USA','Vaccine industry','antivaxxers','big pharma',
                           'conspiracy theory believers','drug addicts','drug users, drunk drivers','government','healthcare system','hospital patients',
                           'hospitals','insurance companies','leftists','media','medical schools','medics','meth addicts',
                           'obese people','obese people, smokers','old man','people','people with pronouns in bios','politicians','pro-restriction',
                           'right wingers','scientists','she','sheeple','sick people','smokers', 'smokers and alcoholics',
                           'social media','some people','the elderly','the elites','the public','the rich','the unvaccinated','the vaccinated','the world today','they']
    elif datasets_singles_us2016:
        list_targets = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
                           'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
                           'NATO', 'Republicans', 'Government','O Malley',
                            'BLM', 'Chafee', 'CNN', 'Chafee', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
                           'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin', 'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly',
                           'Carson', 'Facebook', 'Walker', 'Left', 'Conservatives', 'Trudeau',
                           'Bill Clinton', 'Holt', 'McCain', 'Supporters Trump', 'Biden', 'GOP', 'Kaplan', 'Media', 'Occupy', "Rosie O'Donnell"]
    elif datasets_singles_conspiracy:
        list_targets = ['Americans','BBC','CNN','Dead mother','Fauci',
                    "HolidayOk4857's husband",'JamesHollywoodSEA’s wife','MSM','New York','Not-high school graduates',
                    'People for mandatory vaccination','People taking joy in someone’s death','Pro-vaccinators','Reddit users','Reuters','Russians','Talkradio',
                    'The experts',"TheUnwillingOne's family",'Trump', 'USA','Vaccine industry','antivaxxers','big pharma',
                    'conspiracy theory believers','drug addicts','drug users, drunk drivers','government','healthcare system','hospital patients',
                    'hospitals','insurance companies','leftists','media','medical schools','medics','meth addicts',
                    'obese people','obese people, smokers','old man','people','people with pronouns in bios','politicians','pro-restriction',
                    'right wingers','scientists','she','sheeple','sick people','smokers', 'smokers and alcoholics',
                    'social media','some people','the elderly','the elites','the public','the rich',
                    'the unvaccinated','the vaccinated','the world today','they']
    if datasets_singles_hansard_ethos and (datasets_singles_conspiracy or datasets_singles_us2016):
        list_targets.extend(etho.Target.unique())
    #list_targets = df["Target"].unique()
    #list_targets = [x for x in list_targets if str(x) != "nan"]

    selected_target = st.selectbox("Choose a target entity you would like to analyse", list_targets)

    # all df targets
    df_target_all = pd.DataFrame(df[df.ethos_name != 'neutral']['ethos_name'].value_counts(normalize = True).round(2)*100)
    df_target_all.columns = ['percentage']
    df_target_all.reset_index(inplace=True)
    df_target_all.columns = ['label', 'percentage']
    df_target_all = df_target_all.sort_values(by = 'label')

    df_target_all_att = df_target_all[df_target_all.label == 'attack']['percentage'].iloc[0]
    df_target_all_sup = df_target_all[df_target_all.label == 'support']['percentage'].iloc[0]

    # chosen target df
    df_target = pd.DataFrame(df[df.Target == str(selected_target)]['ethos_name'].value_counts(normalize = True).round(2)*100)
    df_target.columns = ['percentage']
    df_target.reset_index(inplace=True)
    df_target.columns = ['label', 'percentage']

    hero_labels = {'attack', 'support'}
    if len(df_target) == 1:
      if not ("attack" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["attack", 0]
      elif not ("support" in df_target.label.unique()):
          df_target.loc[len(df_target)] = ["support", 0]

    df_target = df_target.sort_values(by = 'label')
    df_target_att = df_target[df_target.label == 'attack']['percentage'].iloc[0]
    df_target_sup = df_target[df_target.label == 'support']['percentage'].iloc[0]


    with st.container():
        st.info(f'Selected entity: *{selected_target}*')
        add_spacelines(1)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Hero score 🙂 👑")
            col1.metric(str(selected_target), f"{len(df[ (df.Target == str(selected_target)) & (df['ethos_name'] == 'support') ])} - " + str(df_target_sup)+ str('%'),
            str(round((df_target_sup - df_target_all_sup),  1))+ str(' p.p.'),
            help = f"Number - percentage of texts that support *{selected_target}*") # round(((df_target_sup / df_target_all_sup) * 100) - 100, 1)

        with col2:
            st.subheader("Anti-hero score 😬 👎")
            col2.metric(str(selected_target), f"{len(df[ (df.Target == str(selected_target)) & (df['ethos_name'] == 'attack') ])} - " + str(df_target_att)+ str('%'),
            str(round((df_target_att - df_target_all_att),  1))+ str(' p.p.'), delta_color="inverse",
            help = f"Number - percentage of texts that attack *{selected_target}*") # ((df_target_att / df_target_all_att) * 100) - 100, 1)

        if not datasets_singles_hansard_ethos:
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
            radio_senti_target = st.radio("Choose the unit of y-axis", ("percentage", "number"))
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)

            df_tar_emo_exp = df[df.Target == selected_target]
            df_tar_emo_exp_senti = df_tar_emo_exp.groupby(['expressed_sentiment'], as_index=False).size()
            df_tar_emo_exp_senti.sort_values(by = 'expressed_sentiment')
            if radio_senti_target == "percentage":
                df_tar_emo_exp_senti['size'] = round(df_tar_emo_exp_senti['size'] / len(df_tar_emo_exp), 3) * 100
            df_tar_emo_exp_senti['expressed_sentiment'] = df_tar_emo_exp_senti['expressed_sentiment'].str.lower()

            user_exp_labs = df_tar_emo_exp_senti['expressed_sentiment'].unique()
            if not ('negative' in user_exp_labs):
                df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['negative', 0]
            if not ('neutral' in user_exp_labs):
                df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['neutral', 0]
            if not ('positive' in user_exp_labs):
                df_tar_emo_exp_senti.loc[len(df_tar_emo_exp_senti)] = ['positive', 0]

            figsenti_user, axsenti = plt.subplots(figsize=(8, 5))
            axsenti.bar(df_tar_emo_exp_senti['expressed_sentiment'], df_tar_emo_exp_senti['size'], color = ['#BB0000', '#022D96', '#026F00'])
            plt.xticks(fontsize=13)
            plt.title(f"Sentiment expressed towards *{selected_target}*\n", fontsize=15)
            vals_senti = df_tar_emo_exp_senti['size'].values.round(1)
            if radio_senti_target == "percentage":
                plt.yticks(np.arange(0, 105, 10), fontsize=12)
                plt.ylabel('percentage %\n', fontsize=13)
                for index_senti, v in enumerate(vals_senti):
                    plt.text(x=index_senti , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=12, ha='center'))
            else:
                if len(df_tar_emo_exp) > 120:
                    plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+16, 20), fontsize=12)
                elif len(df_tar_emo_exp) > 40 and len(df_tar_emo_exp) < 120:
                    plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+6, 5), fontsize=12)
                else:
                    plt.yticks(np.arange(0, df_tar_emo_exp_senti['size'].max()+3, 2), fontsize=12)
                plt.ylabel('number\n', fontsize=13)
                for index_senti, v in enumerate(vals_senti):
                    plt.text(x=index_senti , y = v , s=f"{v}" , fontdict=dict(fontsize=12, ha='center'))
            plt.show()
            st.pyplot(figsenti_user)

    add_spacelines(4)
    with st.expander("(Anti)Hero scores"):
        add_spacelines(1)
        st.write("""
        Hero and Anti-hero scores are calculated based on the ethos annotation. \n

        Values indicate the proportion of social media posts that support and attack, respectively, a given entity.
        Higher Hero score means that users support the target entity more often, and when Anti-hero score is higher, users tend to attack rather than support the entity.
        """)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def UserRhetStrategy():
    #st.header(f" User-Level Analytics ")
    st.subheader(f"LEP Strategies")
    add_spacelines(3)
    plot_type_strategy = st.radio("Choose the type of plot", ('heatmap', 'histogram'))
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']
    if datasets_singles_hansard_ethos:
        df['pathos_name'] = 'neutral'
        df['logos_name'] = 'neutral'
    elif datasets_singles_hansard_logos:
        df['pathos_name'] = 'neutral'
        df['ethos_name'] = 'neutral'
    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
              'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
              'logos_n', 'logos_support_n', 'logos_attack_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)
    if datasets_singles_hansard_ethos:
        cols_strat = ['ethos_support_percent', 'ethos_attack_percent']
    elif datasets_singles_hansard_logos:
        cols_strat = ['logos_support_percent', 'logos_attack_percent']
    else:
        cols_strat = ['ethos_support_percent', 'ethos_attack_percent',
                  'pathos_positive_percent',  'pathos_negative_percent',
                  'logos_support_percent', 'logos_attack_percent']
    if plot_type_strategy == 'histogram':
        def plot_strategies(data):
            i = 0
            for c in range(3):
                print(cols_strat[c+i], cols_strat[c+i+1])
                fig_stats, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
                axs[0].hist(data[cols_strat[c+i]], color='#009C6F')
                title_str0 = " ".join(cols_strat[c+i].split("_")[:-1]).capitalize()
                axs[0].set_title(title_str0, fontsize=20)
                axs[0].set_ylabel('number of users\n', fontsize=15)
                axs[0].set_xlabel('\npercentage of texts %', fontsize=15)
                axs[0].set_xticks(np.arange(0, 101, 10), fontsize=14)

                axs[1].hist(data[cols_strat[c+i+1]], color='#9F0155')
                title_str1 = " ".join(cols_strat[c+i+1].split("_")[:-1]).capitalize()
                axs[1].set_xlabel('\npercentage of texts %', fontsize=15)
                axs[1].yaxis.set_tick_params(labelbottom=True)
                axs[1].set_title(title_str1, fontsize=20)
                axs[1].set_xticks(np.arange(0, 101, 10), fontsize=14)
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
        dimensions = ['ethos_support_percent', 'pathos_positive_percent', 'logos_support_percent']
        if datasets_singles_hansard_ethos:
            dimensions = dimensions[:1]
        elif datasets_singles_hansard_logos:
            dimensions = dimensions[2:]
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
        dimensions_at = ['ethos_attack_percent', 'pathos_negative_percent', 'logos_attack_percent']
        if datasets_singles_hansard_ethos:
            dimensions_at = dimensions_at[:1]
        elif datasets_singles_hansard_logos:
            dimensions_at = dimensions_at[2:]
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

    if not (datasets_singles_hansard_ethos or datasets_singles_hansard_logos):
        ethos_strat = user_stats_df[(user_stats_df.ethos_percent > user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
                (user_stats_df.logos_percent < user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

        pathos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent > user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
                (user_stats_df.logos_percent < user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

        logos_strat = user_stats_df[(user_stats_df.ethos_percent < user_stats_df.ethos_percent.std()+user_stats_df.ethos_percent.mean()) & \
                (user_stats_df.pathos_percent < user_stats_df.pathos_percent.std()+user_stats_df.pathos_percent.mean()) & \
                (user_stats_df.logos_percent > user_stats_df.logos_percent.std()+user_stats_df.logos_percent.mean())]

        with st.container():
            add_spacelines(2)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Dominant logos strategy** 🤔")
                col1.metric(str(logos_strat.shape[0]) + " users", str(round(logos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

            with col2:
                st.write(f"**Dominant ethos strategy** 🗣️👤")
                col2.metric(str(ethos_strat.shape[0]) + " users", str(round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

            with col3:
                st.write(f"**Dominant pathos strategy** ❤️")
                col3.metric(str(pathos_strat.shape[0]) + " users", str(round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1)) + "%")

            add_spacelines(1)
            dominant_percent_strategy = round(pathos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(ethos_strat.shape[0] / len(user_stats_df) * 100, 1) + round(logos_strat.shape[0] / len(user_stats_df) * 100, 1)

            if datasets_singles_conspiracy and datasets_singles_us2016:
                st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant LEP strategy.")
            elif datasets_singles_us2016:
                st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant LEP strategy in *US-Presidential-2016_Reddit* corpus.")
            elif datasets_singles_conspiracy:
                st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant LEP strategy in *Conspiracy-Theories-Vaccines-2021_Reddit* corpus.")
            #st.write(f"##### **{round(dominant_percent_strategy, 1)}%** of users have one dominant LEP strategy in {dataset_name} data.")

    add_spacelines(4)
    with st.expander("LEP strategy"):
        add_spacelines(1)
        st.write("""
        **User LEP strategy**:
        Calculates the proportion of posts generated by a given user that belong to the categories of ethos, pathos, and logos.\n

        **Logos strategy**:
        What percentage of texts generated by a given user attack/support other posts in terms of logos.\n

        **Ethos  strategy**:
        What percentage of texts posted by a given user attack/support other users' or third parties' character.\n\n

        **Pathos strategy**:
        What percentage of texts posted by a given user elicit negative/positive pathos.\n\n

        **Dominant strategy**:
        When a proportion of a user's texts belonging to logos, ethos or pathos is above one standard deviation and a proportion of texts belonging to the other two rhetoric categories is below one standard deviation. \n

        """)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


def ethos_logos_LEPbehaviour_metric(dataframe, source_column = 'Source', labels_column = 'ethos_name'):
  sources_list = dataframe[source_column].unique()
  ethos_metric_value = []
  users_list = []

  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]

    df_user_ethos = df_user.groupby(labels_column, as_index = False)["Text"].size()

    if ('support' in df_user_ethos[labels_column].unique()) or ('attack' in df_user_ethos[labels_column].unique()):
      if 'support' in df_user_ethos[labels_column].unique():
        Nsup_u = df_user_ethos[df_user_ethos[labels_column] == 'support']['size'].iloc[0]
      else:
        Nsup_u = 0
      if 'attack' in df_user_ethos[labels_column].unique():
        Natt_u = df_user_ethos[df_user_ethos[labels_column] == 'attack']['size'].iloc[0]
      else:
        Natt_u = 0
      ethos_user_support = 1 * int(Nsup_u)
      ethos_user_attack = -1 * int(Natt_u)
      ethos_user = (ethos_user_support) + (ethos_user_attack)
      ethos_metric_value.append(ethos_user)
    else:
      ethos_metric_value.append(0)
  df_metric = pd.DataFrame({'source': users_list, 'rhetoric_metric': ethos_metric_value})
  return df_metric

def pathos_LEPbehaviour_metric(dataframe, source_column = 'Source', labels_column = 'pathos_name'):
  sources_list = dataframe[source_column].unique()
  pathos_metric_value = []
  users_list = []

  for u in sources_list:
    users_list.append(str(u))
    df_user = dataframe[dataframe[source_column] == u]

    df_user_pathos = df_user.groupby(labels_column, as_index = False)["Text"].size()

    if ('positive' in df_user_pathos[labels_column].unique()) or ('negative' in df_user_pathos[labels_column].unique()):
      if 'positive' in df_user_pathos[labels_column].unique():
        Nsup_u = df_user_pathos[df_user_pathos[labels_column] == 'positive']['size'].iloc[0]
      else:
        Nsup_u = 0
      if 'negative' in df_user_pathos[labels_column].unique():
        Natt_u = df_user_pathos[df_user_pathos[labels_column] == 'negative']['size'].iloc[0]
      else:
        Natt_u = 0
      pathos_user_support = 1 * int(Nsup_u)
      pathos_user_attack = -1 * int(Natt_u)
      pathos_user = (pathos_user_support) + (pathos_user_attack)
      pathos_metric_value.append(pathos_user)
    else:
      pathos_metric_value.append(0)
  df_metric = pd.DataFrame({'source': users_list, 'rhetoric_metric': pathos_metric_value})
  return df_metric


def UserRhetMetric():
    #st.header(f" User-Level Analytics ")
    st.header(f"LEP Behaviour")
    add_spacelines(2)

    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']

    data_rh = user_rhetoric_v2(df)
    data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]

    user_stats_df = user_stats_app(df)
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
              'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
              'logos_n', 'logos_support_n', 'logos_attack_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)

    color = sns.color_palette("Reds", data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()+15)[::-1][:data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()] +sns.color_palette("Blues", 3)[2:] + sns.color_palette("Greens", data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()+20)[data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()*-1:] # + sns.color_palette("Greens", 15)[4:]

    fig_rh_raw = sns.catplot(kind = 'count', data = data_rh, x = 'rhetoric_metric',
                aspect = 2, palette = color, height = 7)
    for ax in fig_rh_raw.axes.ravel():
      for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.,
            p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
            textcoords = 'offset points')
    plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 30), fontsize=16)
    plt.ylabel('number of users\n', fontsize = 18)
    plt.title("Users LEP behaviour distribution\n", fontsize = 23)
    plt.xticks(fontsize = 16)
    plt.xlabel('\nscore', fontsize = 18)
    plt.show()
    st.pyplot(fig_rh_raw)

    add_spacelines(2)

    # change raw scores to percentages
    counts = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().values
    ids = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().index
    perc = (counts / len(data_rh)) * 100

    data_rh2 = pd.DataFrame({'rhetoric_metric': ids, 'percent':perc})
    data_rh2['percent'] = data_rh2['percent'].apply(lambda x: round(x, 1))

    fig_rh_percent = sns.catplot(kind = 'bar', data = data_rh2, x = 'rhetoric_metric',
                     y = 'percent',
                aspect = 2, palette = color, height = 7, ci = None)
    for ax in fig_rh_percent.axes.ravel():
      for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,
            p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
            textcoords = 'offset points')
        plt.yticks(np.arange(0, data_rh2.percent.max()+6, 5), fontsize = 16)
    plt.ylabel('percentage of users %\n', fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.title("Users LEP behaviour distribution\n", fontsize = 23)
    plt.xlabel('\nscore', fontsize = 18)
    plt.show()
    st.pyplot(fig_rh_percent)

    add_spacelines(4)
    with st.expander("LEP behaviour"):
        add_spacelines(1)
        st.write("""
        Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, and pathos) generated by a given user.
        """)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)

def UsersExtreme():
    st.header("LEP Profiles")
    add_spacelines(2)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    #st.write(f"##### LEP Behaviour")
    if datasets_singles_hansard_logos:
        radio_LEP_behaviour = st.selectbox("Choose the category of a person", ["speakers"])
    else:
        radio_LEP_behaviour = st.selectbox("Choose the category of a person", ("speakers", "target entities"))
    add_spacelines(2)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    radio_LEP_behaviour_axis = st.radio("Choose the unit of y-axis", ("percentage", "number"))
    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']
    if radio_LEP_behaviour == "target entities":
        if datasets_singles_hansard_ethos:
            df['pathos_name'] = 'neutral'
            df['logos_name'] = 'neutral'
            #df[['logos_name', 'ethos_name', 'pathos_name']] = df[['logos_name', 'ethos_name', 'pathos_name']].fillna(0)
        elif datasets_singles_hansard_logos:
            df['pathos_name'] = 'neutral'
            df['ethos_name'] = 'neutral'
            #df[['logos_name', 'ethos_name', 'pathos_name']] = df[['logos_name', 'ethos_name', 'pathos_name']].fillna(0)
        data_rh = user_rhetoric_v2(df, source_column = 'Target')
        user_stats_df = user_stats_app(df, source_column = 'Target')
    else:
        if datasets_singles_hansard_ethos:
            df['pathos_name'] = 'neutral'
            df['logos_name'] = 'neutral'
        elif datasets_singles_hansard_logos:
            df['pathos_name'] = 'neutral'
            df['ethos_name'] = 'neutral'
        data_rh = user_rhetoric_v2(df)
        user_stats_df = user_stats_app(df)

    data_rh = data_rh[ ~(data_rh.user.isin(['[deleted]', 'deleted', 'nan']))]
    user_stats_df.fillna(0, inplace=True)
    for c in ['text_n', 'ethos_n', 'ethos_support_n', 'ethos_attack_n',
              'pathos_n', 'pathos_negative_n', 'pathos_positive_n',
              'logos_n', 'logos_support_n', 'logos_attack_n']:
           user_stats_df[c] = user_stats_df[c].apply(int)

    user_stats_df_desc = user_stats_df.describe().round(3)

    color = sns.color_palette("Reds", data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()+15)[::-1][:data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()] +sns.color_palette("Blues", 3)[2:] + sns.color_palette("Greens", data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()+20)[data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()*-1:] # + sns.color_palette("Greens", 15)[4:]

    if radio_LEP_behaviour_axis == 'number':
        fig_rh_raw = sns.catplot(kind = 'count', data = data_rh, x = 'rhetoric_metric',
                    aspect = 2, palette = color, height = 7)
        for ax in fig_rh_raw.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
        if np.amax(data_rh.rhetoric_metric.value_counts().values) < 50:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+6, 5), fontsize=16)
        elif np.amax(data_rh.rhetoric_metric.value_counts().values) < 400:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 50), fontsize=16)
        else:
            plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+46, 100), fontsize=16)
        plt.ylabel('number of entities\n', fontsize = 18)
        plt.title("LEP behaviour distribution\n", fontsize = 23)
        plt.xticks(fontsize = 16)
        plt.xlabel('\nscore', fontsize = 18)
        plt.show()
        st.pyplot(fig_rh_raw)

    elif radio_LEP_behaviour_axis == 'percentage':
        # change raw scores to percentages
        counts = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().values
        ids = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().index
        perc = (counts / len(data_rh)) * 100

        data_rh2 = pd.DataFrame({'rhetoric_metric': ids, 'percent':perc})
        data_rh2['percent'] = data_rh2['percent'].apply(lambda x: round(x, 1))

        fig_rh_percent = sns.catplot(kind = 'bar', data = data_rh2, x = 'rhetoric_metric',
                         y = 'percent',
                    aspect = 2, palette = color, height = 7, ci = None)
        for ax in fig_rh_percent.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
        if radio_LEP_behaviour == "target entities":
            plt.yticks(np.arange(0, data_rh2.percent.max()+3, 2), fontsize = 16)
        else:
            plt.yticks(np.arange(0, data_rh2.percent.max()+6, 5), fontsize = 16)
        plt.ylabel('percentage of entities %\n', fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.title("LEP behaviour distribution\n", fontsize = 23)
        plt.xlabel('\nscore', fontsize = 18)
        plt.show()
        st.pyplot(fig_rh_percent)

    add_spacelines(3)

    data_rh['standardized_scores'] = standardize(data_rh[['rhetoric_metric']])
    most_neg_users = data_rh.nsmallest(8, 'rhetoric_metric')
    most_pos_users = data_rh.nlargest(8, 'rhetoric_metric')

    most_neg_users_names = most_neg_users.user.tolist()
    most_pos_users_names = most_pos_users.user.tolist()

    if datasets_singles_hansard_logos:
        users_rhet_cols = ['Text', 'logos_name']
    elif datasets_singles_hansard_ethos:
        users_rhet_cols = ['Text', 'Target','ethos_name']
    else:
        users_rhet_cols = ['Text', 'Target', 'pathos_name', 'ethos_name','logos_name']
    with st.container():
        if radio_LEP_behaviour == "target entities":
            head_neg_users = f'<p style="color:#D10000; font-size: 23px; font-weight: bold;">Most negative entities 😈</p>'
        else:
            head_neg_users = f'<p style="color:#D10000; font-size: 23px; font-weight: bold;">Most negative users 😈</p>'
        st.markdown(head_neg_users, unsafe_allow_html=True)
        col111, col222, col333, col444 = st.columns(4)
        with col111:
            st.write(f"**{most_neg_users_names[0]}**")
            col111.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[0], str(round(most_neg_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

        with col222:
            st.write(f"**{most_neg_users_names[1]}**")
            col222.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[1], str(round(most_neg_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

        with col333:
            st.write(f"**{most_neg_users_names[2]}**")
            col333.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[2], str(round(most_neg_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

        with col444:
            st.write(f"**{most_neg_users_names[3]}**")
            col444.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[3], str(round(most_neg_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

    add_spacelines(2)

    with st.container():
        col111, col222, col333, col444 = st.columns(4)
        with col111:
            st.write(f"**{most_neg_users_names[4]}**")
            col111.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[4], str(round(most_neg_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

        with col222:
            st.write(f"**{most_neg_users_names[5]}**")
            col222.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[5], str(round(most_neg_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

        with col333:
            st.write(f"**{most_neg_users_names[6]}**")
            col333.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[6], str(round(most_neg_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

        with col444:
            st.write(f"**{most_neg_users_names[7]}**")
            col444.metric('LEP behaviour score', most_neg_users['rhetoric_metric'].iloc[7], str(round(most_neg_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

        add_spacelines(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if radio_LEP_behaviour == "target entities":
            neg_users_to_df = st.radio("Choose name to see details about the entity \n", most_neg_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"Texts targeted at: **{neg_users_to_df}** ")
            st.dataframe(df[df.Target == str(neg_users_to_df)].set_index("Source")[users_rhet_cols])
        else:
            neg_users_to_df = st.radio("Choose name to see details about the user \n", most_neg_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"Texts posted by: **{neg_users_to_df}** ")
            st.dataframe(df[df.Source == str(neg_users_to_df)].set_index("Source")[users_rhet_cols])
        add_spacelines(1)

    user_stats_df_user1 = user_stats_df[user_stats_df['user'] == str(neg_users_to_df)]
    if not (datasets_singles_hansard_ethos or datasets_singles_hansard_logos):
        with st.container():
            if radio_LEP_behaviour == "target entities":
                st.write(f"##### Users' LEP strategy to speak about {neg_users_to_df}")
            else:
                st.write(f"##### {neg_users_to_df}'s LEP strategy")
            add_spacelines(1)
            col111, col222, col333 = st.columns(3)
            with col111:
                st.write(f"**Logos strategy**")
                col111.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'logos_attack_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

            with col222:
                st.write(f"**Ethos strategy**")
                col222.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_attack_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

            with col333:
                st.write(f"**Pathos strategy**")
                col333.metric(f'{neg_users_to_df}', round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_negative_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'), delta_color="inverse")

            strat_user_val_neg = [round(((user_stats_df_user1['logos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1),
                              round(((user_stats_df_user1['ethos_attack_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1),
                              round(((user_stats_df_user1['pathos_negative_n'] / user_stats_df_user1['text_n']) * 100).iloc[0], 1)]
            strat_user_val_neg_max = np.max(strat_user_val_neg)
            add_spacelines(1)
            strategy_lep_user_neg = []
            if radio_LEP_behaviour == "target entities":
                if strat_user_val_neg[0] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('logos')
                    #st.error(f"Users' negativity in speaking about **{neg_users_to_df}** comes mostly from **logos**")
                if strat_user_val_neg[1] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('ethos')
                    #st.error(f"Users' negativity in speaking about **{neg_users_to_df}** comes mostly from **ethos**")
                if strat_user_val_neg[2] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('pathos')
                strategy_lep_user_neg = " and ".join(strategy_lep_user_neg)
                st.error(f"Users' negativity in speaking about **{neg_users_to_df}** comes mostly from **{strategy_lep_user_neg}**")
            else:
                if strat_user_val_neg[0] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('logos')
                    #st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **logos**")
                if strat_user_val_neg[1] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('ethos')
                    #st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **ethos**")
                if strat_user_val_neg[2] == strat_user_val_neg_max:
                    strategy_lep_user_neg.append('pathos')
                strategy_lep_user_neg = " and ".join(strategy_lep_user_neg)
                st.error(f"**{neg_users_to_df}**'s negativity comes mostly from **{strategy_lep_user_neg}**")

    add_spacelines(1)
    st.write(" **************************************************************************** ")
    add_spacelines(1)

    with st.container():
        if radio_LEP_behaviour == "target entities":
            head_pos_users = f'<p style="color:#00A90D; font-size: 23px; font-weight: bold;">Most positive entities 😀</p>'
        else:
            head_pos_users = f'<p style="color:#00A90D; font-size: 23px; font-weight: bold;">Most positive users 😀</p>'
        st.markdown(head_pos_users, unsafe_allow_html=True)
        col11, col22, col33, col44 = st.columns(4)

        with col11:
            st.write(f"**{most_pos_users_names[0]}**")
            col11.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[0], str(round(most_pos_users['standardized_scores'].iloc[0], 1))+ str(' SD'))

        with col22:
            st.write(f"**{most_pos_users_names[1]}**")
            col22.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[1], str(round(most_pos_users['standardized_scores'].iloc[1], 1))+ str(' SD'))

        with col33:
            st.write(f"**{most_pos_users_names[2]}**")
            col33.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[2], str(round(most_pos_users['standardized_scores'].iloc[2], 1))+ str(' SD'))

        with col44:
            st.write(f"**{most_pos_users_names[3]}**")
            col44.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[3], str(round(most_pos_users['standardized_scores'].iloc[3], 1))+ str(' SD'))

    add_spacelines(2)

    with st.container():
        col11, col22, col33, col44 = st.columns(4)
        with col11:
            st.write(f"**{most_pos_users_names[4]}**")
            col11.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[4], str(round(most_pos_users['standardized_scores'].iloc[4], 1))+ str(' SD'))

        with col22:
            st.write(f"**{most_pos_users_names[5]}**")
            col22.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[5], str(round(most_pos_users['standardized_scores'].iloc[5], 1))+ str(' SD'))

        with col33:
            st.write(f"**{most_pos_users_names[6]}**")
            col33.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[6], str(round(most_pos_users['standardized_scores'].iloc[6], 1))+ str(' SD'))

        with col44:
            st.write(f"**{most_pos_users_names[7]}**")
            col44.metric('LEP behaviour score', most_pos_users['rhetoric_metric'].iloc[7], str(round(most_pos_users['standardized_scores'].iloc[7], 1))+ str(' SD'))

        add_spacelines(2)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        if radio_LEP_behaviour == "target entities":
            pos_users_to_df = st.radio("Choose name to see details about the entity \n", most_pos_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            add_spacelines(1)
            st.write(f"Texts targeted at: **{pos_users_to_df}** ")
            st.dataframe(df[df.Target == str(pos_users_to_df)].set_index("Source")[users_rhet_cols])
            add_spacelines(1)
        else:
            pos_users_to_df = st.radio("Choose name to see details about the user \n", most_pos_users_names)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.write(f"Texts posted by: **{pos_users_to_df}** ")
            st.dataframe(df[df.Source == str(pos_users_to_df)].set_index("Source")[users_rhet_cols])
            add_spacelines(1)

    user_stats_df_user2 = user_stats_df[user_stats_df['user'] == str(pos_users_to_df)]
    if not (datasets_singles_hansard_ethos or datasets_singles_hansard_logos):
        with st.container():
            if radio_LEP_behaviour == "target entities":
                st.write(f"##### Users' LEP strategy to speak about {pos_users_to_df}")
            else:
                st.write(f"##### {pos_users_to_df}'s LEP strategy")
            add_spacelines(1)
            col111, col222, col333 = st.columns(3)
            with col111:
                st.write(f"**Logos strategy**")
                col111.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'logos_support_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

            with col222:
                st.write(f"**Ethos strategy**")
                col222.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'ethos_support_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

            with col333:
                st.write(f"**Pathos strategy**")
                col333.metric(f'{pos_users_to_df}', round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1), str(round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0] - (user_stats_df_desc.loc['mean', 'pathos_positive_n'] / user_stats_df_desc.loc['mean', 'text_n']) * 100, 1))+ str(' p.p.'))

            strat_user_val_pos = [round(((user_stats_df_user2['logos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1),
                              round(((user_stats_df_user2['ethos_support_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1),
                              round(((user_stats_df_user2['pathos_positive_n'] / user_stats_df_user2['text_n']) * 100).iloc[0], 1)]
            strat_user_val_pos_max = np.max(strat_user_val_pos)
            add_spacelines(1)
            strategy_lep_user_pos = []
            if radio_LEP_behaviour == "target entities":
                if strat_user_val_pos[0] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('logos')
                    #st.success(f"Users' positivity in speaking about **{pos_users_to_df}** comes mostly from **logos**")
                if strat_user_val_pos[1] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('ethos')
                    #st.success(f"Users' positivity in speaking about **{pos_users_to_df}** comes mostly from **ethos**")
                if strat_user_val_pos[2] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('pathos')
                strategy_lep_user_pos = " and ".join(strategy_lep_user_pos)
                st.success(f"Users' positivity in speaking about **{pos_users_to_df}** comes mostly from **{strategy_lep_user_pos}**")
            else:
                if strat_user_val_pos[0] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('logos')
                    #st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **logos**")
                if strat_user_val_pos[1] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('ethos')
                    #st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **ethos**")
                if strat_user_val_pos[2] == strat_user_val_pos_max:
                    strategy_lep_user_pos.append('pathos')
                strategy_lep_user_pos = " and ".join(strategy_lep_user_pos)
                st.success(f"**{pos_users_to_df}**'s positivity comes mostly from **{strategy_lep_user_pos}**")

    add_spacelines(4)
    with st.expander("Users LEP Profiles"):
        add_spacelines(1)
        st.write("""
        **LEP behaviour**
        Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, and pathos) generated by a given user.

        **Negative and positive users**:
        Negative and positive users are chosen based on LEP behaviour scores. Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, and pathos) generated by a given user.

        Additionally, we convert LEP behaviour scores into standard deviations for the ease of interpretation.\n
        \n

        **User LEP strategy**:
        Calculates the proportion of posts generated by a given user that belong to the categories of ethos, pathos, and logos.
        """)
    #st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)



def plotRhetoricCompare3(data1, data2, data3, rhet_dim_to_plot):
    rhet_dim = str(rhet_dim_to_plot)
    rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')

    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    df_prop3 = pd.DataFrame(data3[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop3.columns = ['percentage']
    df_prop3.reset_index(inplace=True)
    df_prop3.columns = ['label', 'percentage']
    df_prop3 = df_prop3.sort_values(by = 'label')

    fig_stats, axs = plt.subplots(1, 3, figsize=(26, 6), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + f"*{data1['Dataset'].iloc[0]}*"
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(1)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + f"*{data2['Dataset'].iloc[0]}*"
    axs[1].set_title(title_str1, fontsize=20)
    #axs[1].set_ylabel('percentage %\n', fontsize=16)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(1)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[2].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str3 = str(rhet_dim).capitalize() + " in " +  f"*{data3['Dataset'].iloc[0]}*"
    axs[2].set_title(title_str3, fontsize=20)
    #axs[1].set_ylabel('percentage %\n', fontsize=16)
    axs[2].yaxis.set_tick_params(labelbottom=True)
    axs[2].tick_params(axis='x', labelsize=17)
    axs[2].tick_params(axis='y', labelsize=15)
    vals3 = df_prop3['percentage'].values.round(1)
    for i, v in enumerate(vals3):
        axs[2].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    fig_stats.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats)

def plotRhetoricCompare3_finegrained(data1, data2, data3, rhet_dim_to_plot):
    if rhet_dim_to_plot == 'logos':
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = str(rhet_dim_to_plot)
        data1[rhet_dim_var] = data1[rhet_dim_var].map({'CA':'attack', 'RA':'support'})
        data2[rhet_dim_var] = data2[rhet_dim_var].map({'CA':'attack', 'RA':'support'})
        data3[rhet_dim_var] = data3[rhet_dim_var].map({'CA':'attack', 'RA':'support'})
    else:
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    data1 = data1[data1[rhet_dim_var] != 'neutral']
    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')

    data2 = data2[data2[rhet_dim_var] != 'neutral']
    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    data3 = data3[data3[rhet_dim_var] != 'neutral']
    df_prop3 = pd.DataFrame(data3[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop3.columns = ['percentage']
    df_prop3.reset_index(inplace=True)
    df_prop3.columns = ['label', 'percentage']
    df_prop3 = df_prop3.sort_values(by = 'label')
    if hansard_box and radio_targets == "3rd party":
        df_prop3['percentage'] = [0, 0]

    fig_stats2, axs = plt.subplots(1, 3, figsize=(26, 6), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + f"*{data1['Dataset'].iloc[0]}*"
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(0)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + f"*{data2['Dataset'].iloc[0]}*"
    axs[1].set_title(title_str1, fontsize=20)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(0)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[2].bar(df_prop3['label'], df_prop3['percentage'], color = ['#BB0000', '#026F00'])
    title_str3 = str(rhet_dim).capitalize() + " in " + f"*{data3['Dataset'].iloc[0]}*"
    axs[2].set_title(title_str3, fontsize=20)
    axs[2].yaxis.set_tick_params(labelbottom=True)
    axs[2].tick_params(axis='x', labelsize=17)
    axs[2].tick_params(axis='y', labelsize=15)
    vals3 = df_prop3['percentage'].values.round(0)
    for i, v in enumerate(vals3):
        axs[2].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    fig_stats2.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats2)


def plotRhetoricCompare2(data1, data2, rhet_dim_to_plot):
    rhet_dim = str(rhet_dim_to_plot)
    rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')

    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    fig_stats, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + f"*{data1['Dataset'].iloc[0]}*"
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(1)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#022D96', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + f"*{data2['Dataset'].iloc[0]}*"
    axs[1].set_title(title_str1, fontsize=20)
    #axs[1].set_ylabel('percentage %\n', fontsize=16)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(1)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))
    fig_stats.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats)

def plotRhetoricCompare2_finegrained(data1, data2, rhet_dim_to_plot):
    if rhet_dim_to_plot == 'logos':
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = str(rhet_dim_to_plot)
        data1[rhet_dim_var] = data1[rhet_dim_var].map({'CA':'attack', 'RA':'support'})
        data2[rhet_dim_var] = data2[rhet_dim_var].map({'CA':'attack', 'RA':'support'})
    else:
        rhet_dim = str(rhet_dim_to_plot)
        rhet_dim_var = rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    data1 = data1[data1[rhet_dim_var] != 'neutral']
    df_prop1 = pd.DataFrame(data1[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop1.columns = ['percentage']
    df_prop1.reset_index(inplace=True)
    df_prop1.columns = ['label', 'percentage']
    df_prop1 = df_prop1.sort_values(by = 'label')
    if hansard_box and radio_targets == "3rd party":
        df_prop1['percentage'] = [0, 0]

    data2 = data2[data2[rhet_dim_var] != 'neutral']
    df_prop2 = pd.DataFrame(data2[rhet_dim_var].value_counts(normalize=True).round(3)*100)
    df_prop2.columns = ['percentage']
    df_prop2.reset_index(inplace=True)
    df_prop2.columns = ['label', 'percentage']
    df_prop2 = df_prop2.sort_values(by = 'label')

    fig_stats2, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True, constrained_layout=True)
    axs[0].bar(df_prop1['label'], df_prop1['percentage'], color = ['#BB0000', '#026F00'])
    title_str0 = str(rhet_dim).capitalize() + " in " + f"*{data1['Dataset'].iloc[0]}*"
    axs[0].set_title(title_str0, fontsize=20)
    axs[0].set_ylabel('percentage %\n', fontsize=16)
    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis='x', labelsize=17)
    axs[0].tick_params(axis='y', labelsize=15)
    vals0 = df_prop1['percentage'].values.round(0)
    for i, v in enumerate(vals0):
        axs[0].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))

    axs[1].bar(df_prop2['label'], df_prop2['percentage'], color = ['#BB0000', '#026F00'])
    title_str1 = str(rhet_dim).capitalize() + " in " + f"*{data2['Dataset'].iloc[0]}*"
    axs[1].set_title(title_str1, fontsize=20)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].tick_params(axis='x', labelsize=17)
    axs[1].tick_params(axis='y', labelsize=15)
    vals1 = df_prop2['percentage'].values.round(0)
    for i, v in enumerate(vals1):
        axs[1].text(x=i , y = v+1 , s=f"{v}%" , fontdict=dict(fontsize=14, ha='center', weight='bold'))
    fig_stats2.subplots_adjust(hspace=.3)
    plt.show()
    st.pyplot(fig_stats2)


def CompareDatasetsText():
    st.subheader("Compare Multiple Datasets on Rhetoric")
    add_spacelines(2)

    if hansard_box:
        compare_rhet_dim = st.selectbox("Choose a rhetoric category", rhetoric_dims[:-1], index=0)
    else:
        compare_rhet_dim = st.selectbox("Choose a rhetoric category", rhetoric_dims[::-1], index=0)

    add_spacelines(2)


    if hansard_box and compare_rhet_dim == 'logos':
        hansard_df = hansard_df_logos[['logos_name', 'Dataset']]
    elif hansard_box and compare_rhet_dim == 'ethos':
        hansard_df = hansard_df_ethos[['ethos_name', 'Dataset']]

    if (us2016_box and conspiracy_box and hansard_box):
        plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare3_finegrained(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (us2016_box and conspiracy_box and not hansard_box):
        plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and conspiracy_box and not us2016_box):
        plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and us2016_box and not conspiracy_box):
        plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)

def CompareDatasetsHeroes():
    st.subheader("Comparative Corpora Analysis on (Anti)heroes")
    add_spacelines(2)
    if us2016_box and conspiracy_box and not hansard_box:
        targets_all_conspiracy = ['the vaccinated', 'BBC', 'Reuters', 'Talkradio', 'CNN', 'they',
       'government', 'medics', 'USA', 'Fauci','Pro-vaccinators', 'leftists', 'media', 'big pharma',
       'Reddit users', 'the unvaccinated','The experts', 'Russians', 'the public', 'Americans',
       'the elites', 'social media', 'right wingers', 'Trump',
       'obese people', 'people','the elderly', 'scientists', 'drug addicts',
       'hospital patients', 'medical schools', 'hospitals',
       'pro-restriction', 'the rich', 'politicians', 'antivaxxers',
       'insurance companies', 'healthcare system', 'conspiracy theory believers']

        targets_all_us2016 = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
       'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
       'NATO', 'Republicans', 'Government','O Malley',
       'BLM', 'Chafee', 'CNN', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
       'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin',
       'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly','Carson', 'Facebook',
       'Walker', 'Left', 'Conservatives', 'Trudeau','Bill Clinton', 'Holt', 'McCain',
       'Supporters Trump', 'Biden','GOP', 'Kaplan', 'Media', 'Occupy']

        data1 = us2016_df[us2016_df.Target.isin(targets_all_us2016)]
        data2 = conspiracy_df[conspiracy_df.Target.isin(targets_all_conspiracy)]
        title1 = "US-Presidential-2016_Reddit"
        title2 = "Conspiracy-Theories-Vaccines-2021_Reddit"

    elif conspiracy_box and hansard_box and not us2016_box:
        concat_cols = ['Text', 'Target', 'Ethos_Label', 'ethos_name']

        targets_all_conspiracy = ['the vaccinated', 'BBC', 'Reuters', 'Talkradio', 'CNN', 'they',
       'government', 'medics', 'USA', 'Fauci','Pro-vaccinators', 'leftists', 'media', 'big pharma',
       'Reddit users', 'the unvaccinated','The experts', 'Russians', 'the public', 'Americans',
       'the elites', 'social media', 'right wingers', 'Trump',
       'obese people', 'people','the elderly', 'scientists', 'drug addicts',
       'hospital patients', 'medical schools', 'hospitals',
       'pro-restriction', 'the rich', 'politicians', 'antivaxxers',
       'insurance companies', 'healthcare system', 'conspiracy theory believers']

        data1 = conspiracy_df[conspiracy_df.Target.isin(targets_all_conspiracy)]

        len_targets = hansard_df.groupby("Target", as_index = False).size()
        len_targets = len_targets[len_targets['size'] > 2]
        data2 = hansard_df[hansard_df.Target.isin(len_targets.Target.unique())]

        title1 = "Conspiracy-Theories-Vaccines-2021_Reddit"
        title2 = "Hansard"

    elif us2016_box and hansard_box and not conspiracy_box:
        concat_cols = ['Text', 'Target', 'Ethos_Label', 'ethos_name']

        targets_all_us2016 = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul',
       'Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
       'NATO', 'Republicans', 'Government','O Malley',
       'BLM', 'Chafee', 'CNN', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
       'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin',
       'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly','Carson', 'Facebook',
       'Walker', 'Left', 'Conservatives', 'Trudeau','Bill Clinton', 'Holt', 'McCain',
       'Supporters Trump', 'Biden','GOP', 'Kaplan', 'Media', 'Occupy']

        data1 = us2016_df[us2016_df.Target.isin(targets_all_us2016)]

        len_targets = hansard_df.groupby("Target", as_index = False).size()
        len_targets = len_targets[len_targets['size'] > 2]
        data2 = hansard_df[hansard_df.Target.isin(len_targets.Target.unique())]

        title1 = "US-Presidential-2016_Reddit"
        title2 = "Hansard"

    dd = pd.DataFrame(data1.groupby(['Target'])['Ethos_Label'].value_counts(normalize=True))
    dd.columns = ['normalized_value']
    dd = dd.reset_index()
    dd = dd[dd.Ethos_Label != 0]
    dd_hero = dd[dd.Ethos_Label == 1]
    dd_antihero = dd[dd.Ethos_Label == 2]
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
    dd2['Ethos_Label'] = np.where(dd2.score < 0, 'anti-hero', 'neutral')
    dd2['Ethos_Label'] = np.where(dd2.score > 0, 'hero', dd2['Ethos_Label'])
    dd2 = dd2.sort_values(by = ['Ethos_Label', 'Target'])
    dd2['score'] = dd2['score'] * 100
    dd2['score'] = np.where(dd2.score == 0, 2, dd2['score'])
    dd2_hist['anti hero score'] = dd2anti_scores
    dd2_hist['hero score'] = dd2hero_scores

    dd_conspiracy = pd.DataFrame(data2.groupby(['Target'])['Ethos_Label'].value_counts(normalize=True))
    dd_conspiracy.columns = ['normalized_value']
    dd_conspiracy = dd_conspiracy.reset_index()
    dd_conspiracy = dd_conspiracy[dd_conspiracy.Ethos_Label != 0]
    dd_conspiracy_hero = dd_conspiracy[dd_conspiracy.Ethos_Label == 1]
    dd_conspiracy_antihero = dd_conspiracy[dd_conspiracy.Ethos_Label == 2]

    dd_conspiracy2 = pd.DataFrame({'Target': dd_conspiracy.Target.unique()})
    dd_consp_hist = dd_conspiracy2.copy()
    dd_conspanti_scores = []
    dd_consphero_scores = []

    dd_conspiracy2['score'] = np.nan
    for t in dd_conspiracy.Target.unique():
        try:
            h = dd_conspiracy_hero[dd_conspiracy_hero.Target == t]['normalized_value'].iloc[0]
        except:
            h = 0
        try:
            ah = dd_conspiracy_antihero[dd_conspiracy_antihero.Target == t]['normalized_value'].iloc[0]
        except:
            ah = 0
        dd_consphero_scores.append(h)
        dd_conspanti_scores.append(ah)
        i = dd_conspiracy2[dd_conspiracy2.Target == t].index
        dd_conspiracy2.loc[i, 'score'] = h - ah
    dd_consp_hist['anti hero score'] = dd_conspanti_scores
    dd_consp_hist['hero score'] = dd_consphero_scores

    dd_conspiracy2['Ethos_Label'] = np.where(dd_conspiracy2.score < 0, 'anti-hero', 'neutral')
    dd_conspiracy2['Ethos_Label'] = np.where(dd_conspiracy2.score > 0, 'hero', dd_conspiracy2['Ethos_Label'])
    dd_conspiracy2 = dd_conspiracy2.sort_values(by = ['Ethos_Label', 'Target'])
    dd_conspiracy2['score'] = dd_conspiracy2['score'] * 100
    dd_conspiracy2['score'] = np.where(dd_conspiracy2.score == 0, 2, dd_conspiracy2['score'])

    dd2['Dataset'] = str(title1)
    dd2_hist['Dataset'] = str(title1)
    dd_conspiracy2['Dataset'] = str(title2)
    dd_consp_hist['Dataset'] = str(title2)

    targets_hansard = dd_conspiracy2.Target.unique()
    num_targets_hansard = len(targets_hansard)
    targets_hansard_1 = targets_hansard[:int(num_targets_hansard/2)]
    targets_hansard_2 = targets_hansard[int(num_targets_hansard/2):]

    color = sns.color_palette("Reds", 5)[-1:]  + sns.color_palette("Greens", 5)[::-1][:1] +  sns.color_palette("Blues", 5)[::-1][:1]

    if us2016_box and conspiracy_box and not hansard_box:
        #dd2 = pd.concat([dd2, dd_conspiracy2], axis = 0)
        sns.set(font_scale=2)
        f1 = sns.catplot(kind = 'bar', data = dd2, y = 'Target', x = 'score',
                       hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                       aspect = 1, height = 23, alpha = 1, legend = False, col = "Dataset")
        #plt.xticks(np.arange(-100, 101, 20), fontsize=16)
        #plt.yticks(fontsize=16)
        #plt.xlabel("\nscore", fontsize=18)
        plt.ylabel("")
        f1.set_axis_labels('\nscore', '', fontsize=22)
        plt.legend(fontsize=35, title = '', bbox_to_anchor=(0.8, 1.12), ncol = 3)
        plt.tight_layout()
        sns.set(font_scale=2)
        plt.show()

        f2 = sns.catplot(kind = 'bar', data = dd_conspiracy2,
                       y = 'Target', x = 'score',
                       hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                       aspect = 1, height = 22, alpha = 1, legend = False, col = "Dataset")
        plt.ylabel("")
        f2.set_axis_labels('\nscore', '', fontsize=22)
        plt.legend(fontsize=35, title = '', bbox_to_anchor=(0.8, 1.12), ncol = 3)
        plt.tight_layout()
        sns.set(font_scale=2)
        plt.show()

        plot1, plot2 = st.columns(2)
        with plot1:
            st.pyplot(f1)
        with plot2:
            st.pyplot(f2)

    else:
        sns.set(font_scale=2)
        f1 = sns.catplot(kind = 'bar', data = dd2, y = 'Target', x = 'score',
                       hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                       aspect = 1, height = 23, alpha = 1, legend = False, col = "Dataset")
        #plt.xticks(np.arange(-100, 101, 20), fontsize=16)
        #plt.yticks(fontsize=16)
        #plt.xlabel("\nscore", fontsize=18)
        plt.ylabel("")
        f1.set_axis_labels('\nscore', '', fontsize=22)
        plt.legend(fontsize=35, title = '', bbox_to_anchor=(0.8, 1.12), ncol = 3)
        plt.tight_layout()
        sns.set(font_scale=2)
        plt.show()

        f2 = sns.catplot(kind = 'bar', data = dd_conspiracy2[dd_conspiracy2.Target.isin(targets_hansard_1)],
                       y = 'Target', x = 'score',
                       hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                       aspect = 1, height = 22, alpha = 1, legend = False, col = "Dataset")
        plt.ylabel("")
        f2.set_axis_labels('\nscore', '', fontsize=22)
        plt.legend(fontsize=35, title = '', bbox_to_anchor=(0.65, 1.12), ncol = 3)
        plt.tight_layout()
        sns.set(font_scale=2)
        plt.show()

        f3 = sns.catplot(kind = 'bar', data = dd_conspiracy2[dd_conspiracy2.Target.isin(targets_hansard_2)],
                       y = 'Target', x = 'score',
                       hue = 'Ethos_Label', palette = color, dodge = False, sharey=False,
                       aspect = 1, height = 22, alpha = 1, legend = False, col = "Dataset")
        plt.ylabel("")
        f3.set_axis_labels('\nscore', '', fontsize=22)
        plt.legend(fontsize=35, title = '', bbox_to_anchor=(0.8, 1.12), ncol = 3)
        plt.tight_layout()
        sns.set(font_scale=2)
        plt.show()
        #st.pyplot(f1)
        #st.pyplot(f2)
        #st.pyplot(f3)

        plot1, plot2 = st.columns(2)
        with plot1:
            st.pyplot(f1)
        with plot2:
            st.pyplot(f2)
        plot0, plot3 = st.columns(2)
        with plot0:
            st.write("")
        with plot3:
            st.pyplot(f3)


    add_spacelines(3)

    st.write(f"##### *{title1}*")
    col1, col00, col2 = st.columns([3, 1, 3])
    with col1:
        col1.metric('Heroes 🙂 👑', str(round((len(dd2[dd2['Ethos_Label'] == 'hero']) / len(dd2)) * 100, 1)) + "% ", str(round(round((len(dd2[dd2['Ethos_Label'] == 'hero']) / len(dd2)) * 100, 1) - round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'hero']) / len(dd_conspiracy2)) * 100, 1), 1)) + " p.p. ",
        help = f"Percentage of enities that are viewed as heroes in *{title1}* corpus.")
    with col00:
        st.write("")
    with col2:
        col2.metric('Anti-heroes 😬 👎', str(round((len(dd2[dd2['Ethos_Label'] == 'anti-hero']) / len(dd2)) * 100, 1)) + "% ", str(round(round((len(dd2[dd2['Ethos_Label'] == 'anti-hero']) / len(dd2)) * 100, 1) - round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'anti-hero']) / len(dd_conspiracy2)) * 100, 1), 1)) + " p.p. ",
        delta_color="inverse", help = f"Percentage of enities that are viewed as anti-heroes in *{title1}* corpus.")

    add_spacelines(2)

    st.write(f"##### *{title2}*")
    #st.markdown(f"*{title2}*", unsafe_allow_html=True)
    col101, col0, col202 = st.columns([3, 1, 3])
    with col101:
        col101.metric('Heroes 🙂 👑', str(round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'hero']) / len(dd_conspiracy2)) * 100, 1)) + "% ", str(round(round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'hero']) / len(dd_conspiracy2)) * 100, 1) - round((len(dd2[dd2['Ethos_Label'] == 'hero']) / len(dd2)) * 100, 1), 1)) + " p.p. ",
        help = f"Percentage of enities that are viewed as heroes in *{title2}* corpus.")
    with col0:
        st.write("")
    with col202:
        col202.metric('Anti-heroes 😬 👎', str(round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'anti-hero']) / len(dd_conspiracy2)) * 100, 1)) + "% ", str(round(round((len(dd_conspiracy2[dd_conspiracy2['Ethos_Label'] == 'anti-hero']) / len(dd_conspiracy2)) * 100, 1) - round((len(dd2[dd2['Ethos_Label'] == 'anti-hero']) / len(dd2)) * 100, 1), 1)) + " p.p. ",
        delta_color="inverse", help = f"Percentage of enities that are viewed as anti-heroes in *{title2}* corpus.")






style_css(r"C:\Users\User\Downloads\LEP_test-main\multi_style.css")


#  *********************** sidebar  *********************
with st.sidebar:
    st.write('<style>div[class="css-1siy2j7 e1fqkh3o3"] > div{background-color: #d2cdcd;}</style>', unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)
    st.title("Contents")
    add_spacelines(1)
    contents_radio = st.radio("Pages", ("Main Page", "Single Corpus Analysis", "Comparative Corpora Analysis", "Explore Corpora"))
    #add_spacelines(1)


if contents_radio == 'Comparative Corpora Analysis':
    with st.sidebar:
        add_spacelines(1)
        st.write("**Choose corpora**")
        us2016_box = st.checkbox("US-Presidential-2016_Reddit", value=True)
        conspiracy_box = st.checkbox("Conspiracy-Theories-Vaccines-2021_Reddit", value=True)
        hansard_box = st.checkbox("Hansard_1979-1990")
        if us2016_box:
            us2016_df = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_US2016.xlsx", index_col = 0)
            us2016_df['Dataset'] = "US-Presidential-2016_Reddit"

        if conspiracy_box:
            conspiracy_df = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_conspiracy.xlsx", index_col = 0)
            conspiracy_df['Dataset'] = "Conspiracy-Theories-Vaccines-2021_Reddit"

        if hansard_box:
            hansard_df_ethos = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_hansard_ethos.xlsx", index_col = 0)
            hansard_df_logos = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_hansard_logos.xlsx", index_col = 0)
            hansard_df_ethos['Dataset'] = "Hansard"
            hansard_df_logos['Dataset'] = "Hansard"

        if np.sum([int(us2016_box), int(conspiracy_box), int(hansard_box)]) < 2:
            add_spacelines(1)
            st.warning("**You need to select 2 corpora.**")
            add_spacelines(1)
        #add_spacelines(1)
        #st.write("*******************************************")
        #add_spacelines(1)
        #st.write("## Analysis Unit")
        add_spacelines(1)
        contents_radio_unit2 = st.radio("Analysis Unit",
            ("Text-Based Analysis", 'Person-Based Analysis'))
        #add_spacelines(1)
        #st.write("*******************************************")
        #add_spacelines(1)
        #st.subheader("Analytics Module")
        if contents_radio_unit2 == 'Text-Based Analysis':
            contents_radio3 = st.radio("Analytics Module", ("LEP Distribution", 'WordCloud'))
        elif contents_radio_unit2 == 'Person-Based Analysis':
            contents_radio3 = st.radio("Analytics Module", ['(Anti)Heroes'])
        add_spacelines(1)

elif contents_radio == "Main Page":
    with st.sidebar:
        add_spacelines(24)


elif contents_radio == "Single Corpus Analysis":
    # sidebar
    with st.sidebar:
        add_spacelines(1)
        st.write("**Choose corpora**")
        if 'Hansard_Logos' not in st.session_state and "Hansard_Ethos" not in st.session_state:
            st.session_state['Hansard_Logos'] = False
            st.session_state['Hansard_Ethos'] = False
        datasets_singles_us2016 = st.checkbox("US-Presidential-2016_Reddit", value=True)
        datasets_singles_conspiracy = st.checkbox("Conspiracy-Theories-Vaccines-2021_Reddit")
        datasets_singles_hansard_logos = st.checkbox("Hansard_Logos_1979-1990", disabled=st.session_state.Hansard_Ethos, key="Hansard_Logos")
        datasets_singles_hansard_ethos = st.checkbox("Hansard_Ethos_1979-1990", disabled=st.session_state.Hansard_Logos, key="Hansard_Ethos")
        add_spacelines(1)

        if datasets_singles_conspiracy and datasets_singles_us2016 and not (datasets_singles_hansard_logos or datasets_singles_hansard_ethos):
            concat_cols = ['map_ID', 'Text', 'Source', 'Target', 'No_Ethos', 'Contains_ethos',
                       'Support', 'Attack', 'Ethos_Label', 'No_pathos', 'Contains_pathos',
                       'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
                        'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'RA_logos',
                        'RA_relation', 'CA_logos', 'CA_relation', 'T5_emotion', 'expressed_sentiment',
                        'pathos_name', 'ethos_name', 'logos_name', 'clean_Text', 'clean_Text_lemmatized']
            df01 = load_dataset("US-Presidential-2016_Reddit")
            df02 = load_dataset("Conspiracy-Theories-Vaccines-2021_Reddit")
            df = pd.concat([df01[concat_cols], df02[concat_cols]], axis = 0)
            df = df.reset_index(drop=True)
        elif datasets_singles_conspiracy:
            df = load_dataset("Conspiracy-Theories-Vaccines-2021_Reddit")
        elif datasets_singles_us2016:
            df = load_dataset("US-Presidential-2016_Reddit")
        elif datasets_singles_hansard_logos and not (datasets_singles_us2016 and datasets_singles_conspiracy):
            df = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\app_hansard_logos.xlsx', index_col=0)
        elif datasets_singles_hansard_ethos and not (datasets_singles_us2016 and datasets_singles_conspiracy):
            df = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\app_hansard_ethos.xlsx', index_col=0)
        if datasets_singles_hansard_logos:
            logo = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\app_hansard_logos.xlsx', index_col=0)
            cols_select = ['map_ID', 'Text', 'Source', 'logos_name', 'clean_Text_lemmatized']
            df = pd.concat([df[cols_select], logo[cols_select]], axis = 0)
            df = df.reset_index(drop=True)
        if datasets_singles_hansard_ethos:
            cols_select = ['map_ID', 'Text', 'Source', 'Target', 'Ethos_Label', 'ethos_name', 'clean_Text_lemmatized']
            etho = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\app_hansard_ethos.xlsx', index_col=0)
            df = pd.concat([df[cols_select], etho[cols_select]], axis = 0)
            df = df.reset_index(drop=True)

        #st.write("*******************************************")
        #add_spacelines(1)
        #st.write("## Analysis Unit")
        add_spacelines(1)
        contents_radio_unit = st.radio("Analysis Unit", ("Text-Based Analysis", "Person-Based Analysis"))

        if contents_radio_unit == "Text-Based Analysis":
            # sidebar
            with st.sidebar:
                #add_spacelines(1)
                #st.write("*******************************************")
                #add_spacelines(1)
                #st.subheader("Analytics Module")
                contents_radio2 = st.radio("Analytics Module",
                    ("LEP Distribution", "WordCloud"))
                add_spacelines(2)
        elif contents_radio_unit == "Person-Based Analysis":
            # sidebar
            with st.sidebar:
                #add_spacelines(1)
                #st.write("*******************************************")
                #add_spacelines(1)
                #st.subheader("Analytics Module")
                if datasets_singles_hansard_logos:
                    contents_radio2 = st.radio("Analytics Module", ("LEP Strategies", 'LEP Behaviour', 'LEP Profiles'))
                else:
                    contents_radio2 = st.radio("Analytics Module", ('(Anti)Heroes',"LEP Strategies", 'LEP Behaviour', 'LEP Profiles'))
                add_spacelines(2)

elif contents_radio == 'Explore Corpora':
    with st.sidebar:
        add_spacelines(1)
        corpora_explore_radio = st.selectbox("Choose corpora", ('US-Presidential-2016_Reddit',"Conspiracy-Theories-Vaccines-2021_Reddit", 'Hansard_Logos_1979-1990', 'Hansard_Ethos_1979-1990'))

        if corpora_explore_radio == 'US-Presidential-2016_Reddit':
            dff = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_US2016.xlsx", index_col = 0)
            dff['Dataset'] = "US-Presidential-2016_Reddit"

        elif corpora_explore_radio == 'Conspiracy-Theories-Vaccines-2021_Reddit':
            dff = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_conspiracy.xlsx", index_col = 0)
            dff['Dataset'] = "Conspiracy-Theories-Vaccines-2021_Reddit"

        elif corpora_explore_radio == 'Hansard_Logos_1979-1990':
            dff = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_hansard_logos.xlsx", index_col = 0)
            dff['Dataset'] = "Hansard"

        elif corpora_explore_radio == 'Hansard_Ethos_1979-1990':
            dff = pd.read_excel(r"C:\Users\User\Downloads\LEP_test-main\app_hansard_ethos.xlsx", index_col = 0)
            dff['Dataset'] = "Hansard"


    st.subheader(f"Explore \*{corpora_explore_radio}\*")
    add_spacelines(3)
    if corpora_explore_radio == 'Hansard_Ethos_1979-1990':
        dff_columns = ['Source', 'Target', 'ethos_name', 'map_ID']
        dff = dff[~(dff.Target.isna())]
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[1:3])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                add_spacelines(1)
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        dff_selected['map_ID'] = dff_selected['map_ID'].apply(int)
        st.dataframe(dff_selected[["Text"]+dff_columns].set_index("Source"))
        add_spacelines(1)
        st.write(f"Number of cases: {len(dff_selected)}.")

    elif corpora_explore_radio == 'Hansard_Logos_1979-1990':
        dff_columns = ['Source', 'logos_name', 'map_ID']
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[:2])
        cols_columns = st.columns(len(select_columns))
        dict_cond = {}
        for n, c in enumerate(cols_columns):
            with c:
                add_spacelines(1)
                cond_col = st.multiselect(f"Choose condition for *{select_columns[n]}*",
                                       (dff[select_columns[n]].unique()), (dff[select_columns[n]].unique()[-1]))
                dict_cond[select_columns[n]] = cond_col
        dff_selected = dff.copy()
        for i, k in enumerate(dict_cond.keys()):
            dff_selected = dff_selected[ dff_selected[str(k)].isin(dict_cond[k]) ]
        add_spacelines(2)
        dff_selected['map_ID'] = dff_selected['map_ID'].apply(int)
        st.dataframe(dff_selected[["Text"]+dff_columns].set_index("Source"))
        add_spacelines(1)
        st.write(f"Number of cases: {len(dff_selected)}.")
    else:
        dff_columns = ['Source', 'Target', 'ethos_name', 'logos_name', 'pathos_name',
                       'happiness', 'anger', 'sadness', 'fear', 'disgust', 'surprise', 'trust', 'anticipation', 'map_ID'] #list(dff.columns)
        select_columns = st.multiselect("Choose columns for specifying conditions", dff_columns, dff_columns[1:3])
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
        dff_selected['map_ID'] = dff_selected['map_ID'].apply(int)
        st.dataframe(dff_selected[["Text"]+dff_columns].set_index("Source"))
        add_spacelines(1)
        st.write(f"Number of cases: {len(dff_selected)}.")



#  *********************** page content  *********************

if contents_radio == "Main Page":
    MainPage()

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == "LEP Distribution":
    basicLEPAn()

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == "WordCloud":
    generateWordCloud()

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == '(Anti)Heroes':
    TargetHeroScores()

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == "LEP Strategies":
    UserRhetStrategy()

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == 'LEP Behaviour':
    #UserRhetMetric()
    st.header(f"LEP Behaviour")
    add_spacelines(2)
    rhetoric_dims = ['logos', 'ethos', 'pathos']
    pathos_cols = ['No_pathos', 'Contains_pathos',
           'positive_valence', 'negative_valence', 'happiness', 'anger', 'sadness',
           'fear', 'disgust', 'surprise', 'trust', 'anticipation']
    if datasets_singles_hansard_ethos:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for LEP behaviour distribution", ['ethos'], index=0)
    elif datasets_singles_hansard_logos:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for LEP behaviour distribution", ['logos'], index=0)
    else:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for LEP behaviour distribution", rhetoric_dims[::-1], index=0)
    compare_rhet_dim_variable = selected_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")
    add_spacelines(1)
    check_rhet_dim = st.radio("Choose the unit of y-axis", ("percentage", "number"))

    if selected_rhet_dim == 'pathos':
        data_rh = pathos_LEPbehaviour_metric(dataframe = df, source_column = 'Source', labels_column = 'pathos_name')
        data_rh = data_rh[ ~(data_rh.source.isin(['[deleted]', 'deleted', 'nan']))]
    else:
        data_rh = ethos_logos_LEPbehaviour_metric(dataframe = df, source_column = 'Source', labels_column = str(compare_rhet_dim_variable))
        data_rh = data_rh[ ~(data_rh.source.isin(['[deleted]', 'deleted', 'nan']))]

    color = sns.color_palette("Reds", data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()+15)[::-1][:data_rh[data_rh.rhetoric_metric < 0]['rhetoric_metric'].nunique()] + sns.color_palette("Blues", 3)[2:] + sns.color_palette("Greens", data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()+20)[data_rh[data_rh.rhetoric_metric > 0]['rhetoric_metric'].nunique()*-1:] # + sns.color_palette("Greens", 15)[4:]
    if check_rhet_dim == 'number':
        fig_rh_raw = sns.catplot(kind = 'count', data = data_rh, x = 'rhetoric_metric',
                    aspect = 2, palette = color, height = 7)
        for ax in fig_rh_raw.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
        plt.yticks(np.arange(0, data_rh.rhetoric_metric.value_counts().iloc[0]+26, 50), fontsize=16)
        plt.ylabel('number of sources (speakers)\n', fontsize = 18)
        plt.title(f"{str(selected_rhet_dim).capitalize()} behaviour distribution\n", fontsize = 23)
        plt.xticks(fontsize = 16)
        plt.xlabel(f'\n{selected_rhet_dim} behaviour score', fontsize = 18)
        plt.show()
        st.pyplot(fig_rh_raw)
    else:
        # change raw scores to percentages
        counts = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().values
        ids = data_rh.groupby('rhetoric_metric')['rhetoric_metric'].size().index
        perc = (counts / len(data_rh)) * 100

        data_rh2 = pd.DataFrame({'rhetoric_metric': ids, 'percent':perc})
        data_rh2['percent'] = data_rh2['percent'].apply(lambda x: round(x, 1))

        fig_rh_percent = sns.catplot(kind = 'bar', data = data_rh2, x = 'rhetoric_metric',
                         y = 'percent', aspect = 2, palette = color, height = 7, ci = None)
        for ax in fig_rh_percent.axes.ravel():
          for p in ax.patches:
            ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,
                p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), fontsize = 14.5,
                textcoords = 'offset points')
            plt.yticks(np.arange(0, data_rh2.percent.max()+6, 5), fontsize = 16)
        plt.ylabel('percentage of sources (speakers) %\n', fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.title(f"{str(selected_rhet_dim).capitalize()} behaviour distribution\n", fontsize = 23)
        plt.xlabel(f'\n{selected_rhet_dim} behaviour score', fontsize = 18)
        plt.show()
        st.pyplot(fig_rh_percent)

    add_spacelines(4)
    with st.expander("LEP behaviour"):
        add_spacelines(1)
        st.write("""
        Scores are calculated based on the number of positive/support and negative/attack posts (in terms of logos, ethos, or pathos) generated by a given source (user/speaker).
        """)

elif contents_radio == "Single Corpus Analysis" and contents_radio2 == 'LEP Profiles':
    UsersExtreme()


elif contents_radio == 'Comparative Corpora Analysis' and contents_radio3 == "WordCloud":

    def wordcloud_lexeme_compare2(dataframes, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'crest'):
      if analysis_for == 'attack':
        #print(f'Analysis for: {analysis_for} ')
        cmap_wordcloud =  'Reds' #gist_heat
        for dataframe in dataframes:
            dataframe['% lexeme'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
      elif analysis_for == 'both':
        #print(f'Analysis for: {analysis_for} ')
        cmap_wordcloud = 'autumn'
        for dataframe in dataframes:
            dataframe['% lexeme'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
      else:
        #print(f'Analysis for: {analysis_for} ')
        for dataframe in dataframes:
            dataframe['% lexeme'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

      dfcloud = pd.DataFrame(columns = dataframes[0].columns)
      dfcloud['corpora_n'] = np.nan
      list_uniq_words = []
      dict_uniq_words = {}
      dict_corpora_name = {}
      for n, dataframe in enumerate(dataframes):
          dict_corpora_name[n] = dataframe["Dataset"].iloc[0]
          dfcloud1 = dataframe[(dataframe['% lexeme'] >= int(lexeme_threshold)) & (dataframe['general #'] > 1) & (dataframe.word.map(len)>3)]
          dfcloud1['corpora_n'] = int(n)
          dfcloud1['word'] = dfcloud1['word'].apply(lambda x: str(x).strip())
          list_uniq_words.extend(dfcloud1['word'].unique())
          dict_uniq_words[str(n)] = set(dfcloud1['word'].unique())
          dfcloud = pd.concat([dfcloud, dfcloud1], axis = 0)

      dfcloud = dfcloud.drop_duplicates()
      dfcloud = dfcloud.reset_index(drop=True)
      list_uniq_words = set(list_uniq_words)
      list_uniq_words_select = []
      for w in list_uniq_words:
          shared_word = []
          for n in range(len(dataframes)):
              if w in dfcloud[dfcloud['corpora_n'] == int(n)]['word'].unique():
                  shared_word.append(1)
              else:
                  shared_word.append(0)
          if np.sum(shared_word) == len(dataframes):
              list_uniq_words_select.append(w)

      dfcloud = dfcloud[dfcloud.word.isin(list_uniq_words_select)]
      n_words = dfcloud['word'].nunique()
      text = []
      for w in dfcloud['word'].unique():
        w = str(w).strip()
        if analysis_for == 'both':
            ids = dfcloud[dfcloud.word == w].index
            n = 0
            for i in ids:
                nn = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
                n += nn
        else:
            ids = dfcloud[dfcloud.word == w].index
            n = 0
            for i in ids:
                nn = dfcloud.loc[i, str(analysis_for)+' #'] #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
                n += int(nn)
        l = np.repeat(w, n)
        text.extend(l)

      dfcloud = dfcloud.sort_values(by = ["corpora_n", "word", "% lexeme"])
      dfcloud = dfcloud.drop(["corpora_n"], axis=1)
      dfcloud = dfcloud.rename(columns={"Dataset": "corpora", "% lexeme": "precision", "general #": "overall #"})
      #dfcloud['corpora_n'] = dfcloud['corpora_n'].map(dict_corpora_name)#.map({0: "US-2016-Presidential", 1: "Conspiracy-Theories-Vaccines", 2: "Hansard"})
      import random
      random.shuffle(text)

      st.write(f"There are {n_words} shared words.")
      figure_cloud = make_word_cloud(" ".join(text), 800, 500, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
      return figure_cloud, dfcloud

    st.subheader("Find Common High Precision Words")
    add_spacelines(2)

    if hansard_box:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims[:-1], index=0)
    else:
        selected_rhet_dim = st.selectbox("Choose a rhetoric category for a WordCloud", rhetoric_dims[::-1], index=0)

    compare_rhet_dim_variable = selected_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")
    add_spacelines(1)

    if hansard_box and selected_rhet_dim == 'logos':
        hansard_df = hansard_df_logos[['logos_name', 'Dataset', 'Text', 'Source', 'clean_Text_lemmatized']]
    elif hansard_box and selected_rhet_dim == 'ethos':
        hansard_df = hansard_df_ethos[['ethos_name', 'Dataset', 'Text', 'Source', 'Target', 'clean_Text_lemmatized']]

    label_cloud = st.radio("Choose a subcategory for a WordCloud", ('attack / negative', 'support / positive', 'both'))

    selected_rhet_dim = selected_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")
    label_cloud = label_cloud.replace("attack / negative", "attack").replace("support / positive", "support")
    add_spacelines(1)
    threshold_cloud = st.slider('Select a precision value (threshold) for a WordCloud', 0, 100, 40)
    st.info(f'Selected precision: **{threshold_cloud}**')

    add_spacelines(1)

    #if st.button('Generate a WordCloud'):
    dataframes_all = []
    if (selected_rhet_dim == 'ethos_name') or (selected_rhet_dim == 'logos_name'):
        if (us2016_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(us2016_df[us2016_df[str(selected_rhet_dim)] == 'neutral'],
         us2016_df[us2016_df[str(selected_rhet_dim)] == 'support'],
         us2016_df[us2016_df[str(selected_rhet_dim)] == 'attack'])
         df_for_wordcloud['Dataset'] = us2016_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
        if (conspiracy_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'neutral'],
         conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'support'],
         conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'attack'])
         df_for_wordcloud['Dataset'] = conspiracy_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
        if (hansard_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(hansard_df[hansard_df[str(selected_rhet_dim)] == 'neutral'],
         hansard_df[hansard_df[str(selected_rhet_dim)] == 'support'],
         hansard_df[hansard_df[str(selected_rhet_dim)] == 'attack'])
         df_for_wordcloud['Dataset'] = hansard_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
    else:
        if (us2016_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(us2016_df[us2016_df[str(selected_rhet_dim)] == 'neutral'],
         us2016_df[us2016_df[str(selected_rhet_dim)] == 'positive'],
         us2016_df[us2016_df[str(selected_rhet_dim)] == 'negative'])
         df_for_wordcloud['Dataset'] = us2016_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
        if (conspiracy_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'neutral'],
         conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'positive'],
         conspiracy_df[conspiracy_df[str(selected_rhet_dim)] == 'negative'])
         df_for_wordcloud['Dataset'] = conspiracy_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
        if (hansard_box):
         df_for_wordcloud = prepare_cloud_lexeme_data(hansard_df[hansard_df[str(selected_rhet_dim)] == 'neutral'],
         hansard_df[hansard_df[str(selected_rhet_dim)] == 'positive'],
         hansard_df[hansard_df[str(selected_rhet_dim)] == 'negative'])
         df_for_wordcloud['Dataset'] = hansard_df['Dataset'].iloc[0]
         dataframes_all.append(df_for_wordcloud)
    try:
        fig_cloud, dfcloud_show = wordcloud_lexeme_compare2(dataframes_all, lexeme_threshold = threshold_cloud, analysis_for = str(label_cloud))
        c1, c2, c3 = st.columns([1, 5, 1])
        with c1:
            add_spacelines(1)
        with c2:
            st.pyplot(fig_cloud)
        with c3:
            add_spacelines(1)

        c1, c2, c3 = st.columns([1, 5, 1])
        with c1:
            add_spacelines(1)
        with c2:
            add_spacelines(3)
            st.dataframe(dfcloud_show)
        with c3:
            add_spacelines(1)

    except ValueError:
        add_spacelines(1)
        st.warning("**You need to decrease the precision value** (no words with the selected threshold).")

    #else:
    #    st.warning("**You need to click the button in order to generate a WordCloud**")

    add_spacelines(4)
    with st.expander("High Precision Words"):
        add_spacelines(1)
        st.write("How accurate we are with finding a text belonging to the chosen category when a particular word is present in the text.")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>', unsafe_allow_html=True)


elif contents_radio == 'Comparative Corpora Analysis' and contents_radio3 == "LEP Distribution":
    st.subheader("Compare Multiple Datasets on Rhetoric")
    add_spacelines(2)

    if hansard_box:
        compare_rhet_dim = st.selectbox("Choose a rhetoric category", rhetoric_dims[:-1], index=0)
    else:
        compare_rhet_dim = st.selectbox("Choose a rhetoric category", rhetoric_dims[::-1], index=0)
    add_spacelines(1)
    if compare_rhet_dim == 'logos':
        check_rhet_dim_unit = st.radio("The unit of text", ["RAs and CAs"])
    elif compare_rhet_dim == 'pathos':
        check_rhet_dim_unit = st.radio("The unit of text", ["ADU"])

    elif compare_rhet_dim == 'ethos' and not hansard_box:
        check_rhet_dim_unit = st.radio("The unit of text", ["ADU"])

    elif compare_rhet_dim == 'ethos' and hansard_box and conspiracy_box and us2016_box:
        cc1, cc2, cc3 = st.columns([3, 3, 2])
        with cc3:
            check_rhet_dim_unitH = st.radio("The unit of text for *Hansard*", ["post"])
        with cc2:
            check_rhet_dim_unitH = st.radio("The unit of text for *Conspiracy-Theories-Vaccines-2021_Reddit*", ["ADU"])
        with cc1:
            check_rhet_dim_unitH = st.radio("The unit of text for *US-Presidential-2016_Reddit*", ["ADU"])

    elif compare_rhet_dim == 'ethos' and hansard_box and us2016_box:
        cc1, cc2 = st.columns(2)
        with cc2:
            check_rhet_dim_unitH = st.radio("The unit of text for *Hansard*", ["post"])
        with cc1:
            check_rhet_dim_unitH = st.radio("The unit of text for *US-Presidential-2016_Reddit*", ["ADU"])

    elif compare_rhet_dim == 'ethos' and hansard_box and conspiracy_box:
        cc1, cc2 = st.columns(2)
        with cc2:
            check_rhet_dim_unitH = st.radio("The unit of text for *Hansard*", ["post"])
        with cc1:
            check_rhet_dim_unitH = st.radio("The unit of text for *Conspiracy-Theories-Vaccines-2021_Reddit*", ["ADU"])

    compare_rhet_dim_variable = compare_rhet_dim.replace("ethos", "ethos_name").replace("logos", "logos_name").replace("pathos", "pathos_name")

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;font-size=18px;}</style>', unsafe_allow_html=True)
    if compare_rhet_dim == 'logos':
        radio_targets = ""
    else:
        radio_targets = st.radio("Choose the target of ethos appeals", ("all types", "direct", "3rd party"))
        add_spacelines(1)

    if hansard_box and compare_rhet_dim == 'logos':
        hansard_df = hansard_df_logos[['logos_name', 'Dataset', 'Text', 'Source', 'clean_Text_lemmatized']]
    elif hansard_box and compare_rhet_dim == 'ethos':
        hansard_df = hansard_df_ethos[['ethos_name', 'Dataset', 'Text', 'Source', 'Target', 'clean_Text_lemmatized']]

    if radio_targets == "direct":
        if us2016_box:
            user2user_targets_us2016 = ['Emerica', 'ESP16', 'PixelsAreYourFriends', 'deleted',
            'broduding', 'AgentMullWork', 'Kagawful', 'BeerFarts86',
            'bashar_al_assad', 'Erra0', 'For_mobile', 'shadowofahelicopter','XxEnder_GamerxX', 'reaper527', 'Seakawn','vectorlit', 'rowdyroddypiperjr', 'Rocag','captaindammit87', 'sddnpoopxplsndss', 'sddnpoopxplsndss',
            'DeceitFive9', 'progress18', 'SapCPark', 'Twer_for_Hillary','thelazt1', 'dj_kled', 'recalcitrantJester', 'OmegaFemale', 'No_stop_signs', 'oldmanbrownsocks', 'Mongfight',
            'kevie3drinks',  'OurBenefactors', '143jammy', '134jammy', 'peyote_the_coyote', 'unknown0', 'unknown1', 'unknown2', 'unknown3', 'unknown4']
            us2016_df = us2016_df[us2016_df.Target.isin(user2user_targets_us2016)]

        if conspiracy_box :
            user2user_targets_conspiracy = ['testtube- accident',  'PopcornFuel','unknown', 'Defiant Dragon', 'Gherin 29',
                                            'LordOFthe Noldor', 'jblank84', 'Meg_119', 'LouisAngel39','Segundaleydenewtonnn','ArmedWithBars', 'PorcelainPoppy', 'BrandonDunarote',
                                            'Wooden-Building', 'testtube-accident','twichy1983', 'chaos_magician_', 'MrGurbic','BuzzedCauldron','Electrical_Scholar40', 'madandwell', 'Effective_Ad4588',
                                            'daydreamerinny', 'PirateShorty', 'JamesHollywoodSEA','Rpowdigs','LicksMackenzie', 'you', 'horror_junkie5919', 'thisbliss8',
                                            'TheUnwillingOne','Legitimate_Ad416', 'xynapse', 'sol_sleepy', 'nleven','ZeerVreemd', 'ShitForBranes', 'olivercorgi7','Icy-Ad-5551', 'SerpentineApotheosis', 'Classic-Heron-1676',
                                            'Professor_Matty', 'maelstrom5', 'previous poster', 'Dregoncrys', 'Hardfloor', '@Domified','Daehoidar', 'R0xx0Rs-Mc0wNaGe', 'TheAutoAlly', 'Pkmntrainer91','HolidayOk4857','Economy-Cut-7355',
                                            'kyfto','u/the_golden_girls', 'us', 'we']
            conspiracy_df = conspiracy_df[conspiracy_df.Target.isin(user2user_targets_conspiracy)]

    elif radio_targets == "3rd party":
        if us2016_box:
            party_targets_us2016 = ['Webb', 'Sanders', 'Clinton', 'Trump', 'Paul','Romney', 'Obama', 'Cooper', 'Democrats', 'Russia',
            'NATO', 'Republicans', 'Government','O Malley','BLM', 'Chafee', 'CNN', 'Chafee', 'Bush', 'Snowden', 'the Times', 'USA', 'Huckabee',
            'Christie', 'Joe Rogan', 'John Oliver', 'Sarah Palin', 'Fox News', 'Kasich', 'Perry', 'Rubio', 'Cruz', 'Kelly','Carson', 'Facebook', 'Walker', 'Left', 'Conservatives', 'Trudeau',
            'Bill Clinton', 'Holt', 'McCain', 'Supporters Trump', 'Biden', 'GOP', 'Kaplan', 'Media', 'Occupy', "Rosie O'Donnell"]
            us2016_df = us2016_df[us2016_df.Target.isin(party_targets_us2016)]

        if conspiracy_box :
            party_targets_conspiracy = ['Americans','BBC','CNN','Dead mother','Fauci',"HolidayOk4857's husband",'JamesHollywoodSEA’s wife','MSM','New York','Not-high school graduates',
            'People for mandatory vaccination','People taking joy in someone’s death','Pro-vaccinators','Reddit users','Reuters','Russians','Talkradio',
            'The experts',"TheUnwillingOne's family",'Trump', 'USA','Vaccine industry','antivaxxers','big pharma','conspiracy theory believers','drug addicts','drug users, drunk drivers','government','healthcare system','hospital patients',
            'hospitals','insurance companies','leftists','media','medical schools','medics','meth addicts','obese people','obese people, smokers','old man','people','people with pronouns in bios','politicians','pro-restriction',
            'right wingers','scientists','she','sheeple','sick people','smokers', 'smokers and alcoholics','social media','some people','the elderly','the elites','the public','the rich',
            'the unvaccinated','the vaccinated','the world today','they']
            conspiracy_df = conspiracy_df[conspiracy_df.Target.isin(party_targets_conspiracy)]


    if (us2016_box and conspiracy_box and hansard_box and radio_targets != "all types" and compare_rhet_dim == 'ethos'):
        #plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
        #add_spacelines(2)
        plotRhetoricCompare3_finegrained(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (us2016_box and conspiracy_box and hansard_box):
        #plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(1)
        if compare_rhet_dim == 'logos':
            dd1 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_US2016.xlsx')
            dd2 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_CT.xlsx')
            dd3 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_hansard.xlsx')
            dd1['Dataset'] = "US-Presidential-2016_Reddit"
            dd2['Dataset'] = "Conspiracy-Theories-Vaccines-2021_Reddit"
            dd3['Dataset'] = "Hansard"
            plotRhetoricCompare3_finegrained(data1 = dd1, data2 = dd2, data3 = dd3, rhet_dim_to_plot = compare_rhet_dim)
        elif compare_rhet_dim == 'pathos':
            plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare3_finegrained(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
            fig_emo_pat1 = plot_pathos_emo(us2016_df, title = '')
            fig_emo_pat2 = plot_pathos_emo(conspiracy_df, title = '')
            coll1, coll2, coll3 = st.columns(3)
            with coll1:
                st.pyplot(fig_emo_pat1)
            with coll2:
                st.pyplot(fig_emo_pat2)
            with coll3:
                st.info("Pathos annotation for *Hansard* corpus is currently not available")
        else:
            plotRhetoricCompare3(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare3_finegrained(data1 = us2016_df, data2 = conspiracy_df, data3 = hansard_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (us2016_box and conspiracy_box and not hansard_box and radio_targets != "all types" and compare_rhet_dim == 'ethos'):
        #plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        #add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (us2016_box and conspiracy_box and not hansard_box):
        #plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(1)
        if compare_rhet_dim == 'logos':
            dd1 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_US2016.xlsx')
            dd2 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_CT.xlsx')
            dd1['Dataset'] = "US-Presidential-2016_Reddit"
            dd2['Dataset'] = "Conspiracy-Theories-Vaccines-2021_Reddit"
            plotRhetoricCompare2_finegrained(data1 = dd1, data2 = dd2, rhet_dim_to_plot = compare_rhet_dim)

        elif compare_rhet_dim == 'pathos':
            plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

            fig_emo_pat1 = plot_pathos_emo(us2016_df, title = '')
            fig_emo_pat2 = plot_pathos_emo(conspiracy_df, title = '')
            coll1, coll2, = st.columns(2)
            with coll1:
                st.pyplot(fig_emo_pat1)
            with coll2:
                st.pyplot(fig_emo_pat2)

        else:
            plotRhetoricCompare2(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = us2016_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and conspiracy_box and not us2016_box and radio_targets != "all types" and compare_rhet_dim == 'ethos'):
        #plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        #add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and conspiracy_box and not us2016_box):
        #plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(1)
        if compare_rhet_dim == 'logos':
            dd1 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_hansard.xlsx')
            dd2 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_CT.xlsx')
            dd1['Dataset'] = "Hansard"
            dd2['Dataset'] = "Conspiracy-Theories-Vaccines-2021_Reddit"
            plotRhetoricCompare2_finegrained(data1 = dd1, data2 = dd2, rhet_dim_to_plot = compare_rhet_dim)

        elif compare_rhet_dim == 'pathos':
            plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
            fig_emo_pat2 = plot_pathos_emo(conspiracy_df, title = '')
            coll1, coll2, = st.columns(2)
            with coll1:
                st.info("Pathos annotation for *Hansard* corpus is currently not available")
            with coll2:
                st.pyplot(fig_emo_pat2)
        else:
            plotRhetoricCompare2(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = conspiracy_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and us2016_box and not conspiracy_box and radio_targets != "all types" and compare_rhet_dim == 'ethos'):
        #plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
        #add_spacelines(2)
        plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)

    elif (hansard_box and us2016_box and not conspiracy_box):
        #plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
        add_spacelines(1)
        if compare_rhet_dim == 'logos':
            dd1 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_hansard.xlsx')
            dd2 = pd.read_excel(r'C:\Users\User\Downloads\LEP_test-main\LOGOS_US2016.xlsx')
            dd1['Dataset'] = "Hansard"
            dd2['Dataset'] = "US-Presidential-2016_Reddit"
            plotRhetoricCompare2_finegrained(data1 = dd1, data2 = dd2, rhet_dim_to_plot = compare_rhet_dim)

        elif compare_rhet_dim == 'pathos':
            plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
            fig_emo_pat2 = plot_pathos_emo(us2016_df, title = '')
            coll1, coll2, = st.columns(2)
            with coll1:
                st.info("Pathos annotation for *Hansard* corpus is currently not available")
            with coll2:
                st.pyplot(fig_emo_pat2)
        else:
            plotRhetoricCompare2(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)
            add_spacelines(2)
            plotRhetoricCompare2_finegrained(data1 = hansard_df, data2 = us2016_df, rhet_dim_to_plot = compare_rhet_dim)

    add_spacelines(3)
    st.write(f"### Qualitative analysis")
    add_spacelines(1)
    if radio_targets != "all types" and compare_rhet_dim == 'ethos':
        category_df_examples = st.selectbox("Choose a category", ["attack/negative", "support/positive"], index=0)
    else:
        category_df_examples = st.selectbox("Choose a category", ["attack/negative", "support/positive", "neutral"], index=0)
    add_spacelines(1)
    if category_df_examples == "attack/negative":
        st.error(f"Displaying posts that are from the category of: **{compare_rhet_dim} {category_df_examples}** (red color bars above).")
    elif category_df_examples == "support/positive":
        st.success(f"Displaying posts that are from the category of: **{compare_rhet_dim} {category_df_examples}** (green color bars above).")
    else:
        st.info(f"Displaying posts that are from the category of: **{compare_rhet_dim} {category_df_examples}** (blue color bars above).")
    add_spacelines(1)
    if (us2016_box and conspiracy_box and hansard_box):
        col1_qual, col2_qual, col3_qual = st.columns(3)
        if category_df_examples == "attack/negative" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                add_spacelines(1)
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                add_spacelines(1)
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'])}.")

        elif category_df_examples == "attack/negative" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                add_spacelines(1)
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                add_spacelines(1)
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")

        elif category_df_examples == "support/positive" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                add_spacelines(1)
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                add_spacelines(1)
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'])}.")

        elif category_df_examples == "support/positive" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                add_spacelines(1)
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                add_spacelines(1)
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")
        else:
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                add_spacelines(1)
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                add_spacelines(1)
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")

    elif (us2016_box and conspiracy_box and not hansard_box):
        col1_qual, col2_qual = st.columns(2)
        if category_df_examples == "attack/negative" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                #add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'])}.")

        elif category_df_examples == "attack/negative" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                #add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'])}.")

        elif category_df_examples == "support/positive" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                #add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'])}.")

        elif category_df_examples == "support/positive" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                #add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
        else:
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                #add_spacelines(1)
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")

    elif (hansard_box and conspiracy_box and not us2016_box):
        col2_qual, col3_qual = st.columns(2)
        if category_df_examples == "attack/negative" and compare_rhet_dim != 'pathos':
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'attack'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'])}.")

        elif category_df_examples == "attack/negative" and compare_rhet_dim == 'pathos':
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'negative'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")

        elif category_df_examples == "support/positive" and compare_rhet_dim != 'pathos':
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'support'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'])}.")

        elif category_df_examples == "support/positive" and compare_rhet_dim == 'pathos':
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")
        else:
            with col2_qual:
                st.write(f"*Conspiracy-Theories-Vaccines-2021_Reddit*")
                st.dataframe(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(conspiracy_df[conspiracy_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")

    elif (hansard_box and us2016_box and not conspiracy_box):
        col1_qual, col3_qual = st.columns(2)
        if category_df_examples == "attack/negative" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'attack'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'attack'])}.")

        elif category_df_examples == "attack/negative" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'negative'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")

        elif category_df_examples == "support/positive" and compare_rhet_dim != 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'support'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'support'])}.")

        elif category_df_examples == "support/positive" and compare_rhet_dim == 'pathos':
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'positive'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                st.warning(f"Pathos annotation is currently not available for the *Hansard* corpus.")
        else:
            with col1_qual:
                st.write(f"*US-Presidential-2016_Reddit*")
                st.dataframe(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                add_spacelines(1)
                st.write(f"Number of examples: {len(us2016_df[us2016_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")
            with col3_qual:
                st.write(f"*Hansard*")
                if hansard_box and radio_targets == "3rd party":
                    st.warning("No examples from the selected category.")
                else:
                    st.dataframe(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'].set_index("Source")[['Text']])
                    add_spacelines(1)
                    st.write(f"Number of examples: {len(hansard_df[hansard_df[str(compare_rhet_dim_variable)] == 'neutral'])}.")



elif contents_radio == 'Comparative Corpora Analysis' and contents_radio3 == "(Anti)Heroes":
    if hansard_box:
        hansard_df = hansard_df_ethos

    if np.sum([int(us2016_box), int(conspiracy_box), int(hansard_box)]) == 2:
        CompareDatasetsHeroes()
    elif np.sum([int(us2016_box), int(conspiracy_box), int(hansard_box)]) == 3:
        st.subheader("Comparative Corpora Analysis on (Anti)heroes")
        add_spacelines(3)
        st.warning("**You can select only 2 corpora to compare (Anti)Heroes.**")
    else:
        st.subheader("Comparative Corpora Analysis on (Anti)heroes")
        add_spacelines(3)
        st.warning("**You need to select 2 corpora.**")
