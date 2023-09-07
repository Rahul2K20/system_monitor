import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import json
sns.set_theme()

def anomaly_detector(lst):
    m, s = np.mean(lst), np.std(lst)
    cutoff = s*2
    lower, upper = m - cutoff, m + cutoff
    anomalies = []
    for v in lst:
      if v <= lower:
        anomalies.append("anomaly")
      elif v >= upper:
        anomalies.append("anomaly")
      else:
        anomalies.append("normal")
    df_anomaly = pd.DataFrame({"f1_score":lst, "anomaly":anomalies}, columns=["f1_score", "anomaly"])
    return df_anomaly

df = pd.read_csv("utilities/runs (5).csv")
df = df.drop(['Duration', 'Run ID', 'Source Type', 'Source Name', 'User', 'Status', 'Name', 'batch_size', 'eval_frequency', 'initial_rate', 'max_steps', 'model_name', 'optimizers', 'section_model', 'size', 'model_flavor'], axis=1)
df_entire_resume = df[df['section'] == "entire_resume"]
df_entire_resume = df_entire_resume.drop(['AWARDS_f1_score', 'AWARDS_precision', 'AWARDS_recall', 'Address_f1_score', 'Address_precision', 'Address_recall', 'GITHUB_URL_f1_score', 'GITHUB_URL_precision', 'GITHUB_URL_recall', 'LinkedIn URL_f1_score', 'LinkedIn URL_precision', 'LinkedIn URL_recall', 'SUMMARY_f1_score', 'SUMMARY_precision', 'SUMMARY_recall'], axis = 1)
df_personal_details_section = df[df['section'] == "personal_details_section"]
df_experience_details_section = df[df['section'] == "experience_details_section"]
df_education_details_section = df[df['section'] == "education_details_section"]

st.set_page_config(page_title="MLOPS Monitoring System Dashboard", layout="wide")
st.title(":bar_chart: MLOPS Monitoring System")

with st.sidebar:
    button_3 = st.button("System Health", type="primary")
    button_1 = st.button("Performance Metrics", type="primary")
    button_2 = st.button("Anomaly Detection", type="primary")
    button_4 = st.button("covariate shift", type="primary")
    button_5 = st.button("univariate shift", type="primary")

if button_1:
    st.write("i have pressed button_1")
    c1, c2 = st.columns((10, 10))
    with c1:
        st.markdown('Entire Resume Performance Metrics')
        f1_score_lst = df_entire_resume['f1_score'].to_list()
        cumulative_f1 = [np.mean(f1_score_lst[:n]) for n in range(1, len(f1_score_lst) + 1)]
        window_size = 5
        sliding_f1 = np.convolve(f1_score_lst, np.ones(window_size) / window_size, mode="valid")
        max_f1 = max(f1_score_lst)
        y = df_entire_resume['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        fig_1 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed",
                   label="threshold")
        plt.plot(cumulative_f1, label="cumulative")
        plt.plot(sliding_f1, label="sliding")
        plt.legend()
        st.pyplot(fig_1)
        st.markdown('Personal Details Performance Metrics')
        f1_score_lst = df_personal_details_section['f1_score'].to_list()
        cumulative_f1 = [np.mean(f1_score_lst[:n]) for n in range(1, len(f1_score_lst) + 1)]
        window_size = 5
        sliding_f1 = np.convolve(f1_score_lst, np.ones(window_size) / window_size, mode="valid")
        max_f1 = max(f1_score_lst)
        y = df_personal_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        fig_2 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed",
                   label="threshold")
        plt.plot(cumulative_f1, label="cumulative")
        plt.plot(sliding_f1, label="sliding")
        plt.legend()
        st.pyplot(fig_2)
    with c2:
        st.markdown('Experience performance Performance Metrics')
        f1_score_lst = df_experience_details_section['f1_score'].to_list()
        cumulative_f1 = [np.mean(f1_score_lst[:n]) for n in range(1, len(f1_score_lst) + 1)]
        window_size = 5
        sliding_f1 = np.convolve(f1_score_lst, np.ones(window_size) / window_size, mode="valid")
        max_f1 = max(f1_score_lst)
        y = df_experience_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        fig_3 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed",
                   label="threshold")
        plt.plot(cumulative_f1, label="cumulative")
        plt.plot(sliding_f1, label="sliding")
        plt.legend()
        st.pyplot(fig_3)
        st.markdown('Education Details Performance Metrics')
        f1_score_lst = df_education_details_section['f1_score'].to_list()
        cumulative_f1 = [np.mean(f1_score_lst[:n]) for n in range(1, len(f1_score_lst) + 1)]
        window_size = 5
        sliding_f1 = np.convolve(f1_score_lst, np.ones(window_size) / window_size, mode="valid")
        max_f1 = max(f1_score_lst)
        y = df_education_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        fig_4 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed",
                   label="threshold")
        plt.plot(cumulative_f1, label="cumulative")
        plt.plot(sliding_f1, label="sliding")
        plt.legend()
        st.pyplot(fig_4)
elif button_2:
    print("i have pressed button_2")
    st.write("i have pressed button_2")
    c1, c2 = st.columns((10, 10))
    with c1:
        import matplotlib.pyplot as plt
        st.markdown('Entire Resume Anomaly')
        y = df_entire_resume['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        anomaly_df = anomaly_detector(df_entire_resume['f1_score'].to_list())
        fig_5 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt = sns.scatterplot(x=anomaly_df.index, y=anomaly_df['f1_score'], hue=anomaly_df['anomaly'])
        st.pyplot(fig_5)
        st.markdown('Personal Details Anomaly')
        y = df_personal_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        import matplotlib.pyplot as plt
        anomaly_df = anomaly_detector(df_personal_details_section['f1_score'].to_list())
        fig_6 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt = sns.scatterplot(x=anomaly_df.index, y=anomaly_df['f1_score'], hue=anomaly_df['anomaly'])
        st.pyplot(fig_6)
    with c2:
        st.markdown('Experience Details Anomaly')
        y = df_experience_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        import matplotlib.pyplot as plt
        anomaly_df = anomaly_detector(df_experience_details_section['f1_score'].to_list())
        fig_7 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt = sns.scatterplot(x=anomaly_df.index, y=anomaly_df['f1_score'], hue=anomaly_df['anomaly'])
        st.pyplot(fig_7)
        st.markdown('Education Details Anomaly')
        y = df_education_details_section['f1_score'].to_list()
        x = [n for n in range(0, len(y))]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        import matplotlib.pyplot as plt
        anomaly_df = anomaly_detector(df_education_details_section['f1_score'].to_list())
        fig_8 = plt.figure()
        plt.plot(x, p(x), label='trendline')
        plt = sns.scatterplot(x=anomaly_df.index, y=anomaly_df['f1_score'], hue=anomaly_df['anomaly'])
        st.pyplot(fig_8)
elif button_3:
        image = Image.open('utilities/system_health.png')
        st.image(image, caption='System health')
elif button_4:
    print("i have pressed button_4")
    st.write("i have pressed button_4")
elif button_5:
    print("i have pressed button_5")
    st.write("i have pressed button_5")
else:
    pass
