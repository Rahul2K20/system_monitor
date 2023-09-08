import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import json

sns.set_theme()

def home():
    st.title("Welcome to Our Resume Analyzer!")
    st.write("A project by People Tech Group to facilitate resume evaluations.")
        
        
    gif_url = "https://www.thedigitalgroup.com/products/digital-resume-parser/assets/img/about/resume-parsing-option.gif"
    st.image(gif_url, caption='Resume Parsing in Action', use_column_width=True)

    st.subheader("Features at a Glance:")
    st.markdown("""
    - **Performance Metrics**: Comprehensive quality and relevance evaluation of entire resumes.
    - **Anomaly Detection**: Pinpoint inconsistencies, backed by an F1 score visualization.
    - **Covariate Shift**: Witness the evolution of data over time through intuitive bar graphs.
    - **Univariate Shift**: Keep an eye on trending resume keywords.
    """)


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

st.set_page_config(page_title="MLOPS Monitoring System Dashboard", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: MLOPS Monitoring System")

imgSource = "https://i.ibb.co/sbGk2NW/People-Tech-Group-001-s-QBw-S61-max-600x480.png"
response = requests.get(imgSource)
img = Image.open(BytesIO(response.content))

st.sidebar.image(img, use_column_width=True)

with st.sidebar:
    button_0 = st.button("Home", type="primary")
    button_3 = st.button("System Health", type="primary")
    button_1 = st.button("Performance Metrics", type="primary")
    button_2 = st.button("Anomaly Detection", type="primary")
    button_4 = st.button("Covariate shift", type="primary")
    button_5 = st.button("Univariate shift", type="primary")
    button_6 = st.button("Concept shift", type="primary")


if button_0:
    home()

else: 
    if button_1:
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
            plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed", label="threshold")
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
            plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed", label="threshold")
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
            plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed", label="threshold")
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
            plt.hlines(y=(max_f1 - 0.03), xmin=0, xmax=len(f1_score_lst), colors="blue", linestyles="dashed", label="threshold")
            plt.plot(cumulative_f1, label="cumulative")
            plt.plot(sliding_f1, label="sliding")
            plt.legend()
            st.pyplot(fig_4)
    elif button_2:
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
        f_1 = open('utilities/dataset_0.6954225242382301.json')
        f_2 = open('utilities/dataset_0.7436912184166055.json')
        data_1 = json.load(f_1)
        data_2 = json.load(f_2)
        count_dict_1 = {}
        for d in data_1:
            temp_entities = d[1]['entities']
            for ent in temp_entities:
                if ent[2] in count_dict_1.keys():
                    count_dict_1[ent[2]] = count_dict_1[ent[2]] + 1
                else:
                    count_dict_1[ent[2]] = 1
        sum_of_entities = sum(count_dict_1.values())
        new_count_dict_1 = {}
        for key in count_dict_1.keys():
            if key == "AWARDS":
                key_m = "AW"
            elif key == "CERTIFICATIONS":
                key_m = "CT"
            elif key == "CLIENT":
                key_m = "CL"
            elif key == "COMPANY":
                key_m = "COM"
            elif key == "COMPANY_LOC":
                key_m = "COM_LOC"
            elif key == "DESGINATION":
                key_m = "DESG"
            elif key == "EDUCATIONAL_INSTITUTION":
                key_m = "E_I"
            elif key == "GITHUB_URL":
                key_m = "G_U"
            elif key == "HIGHER_EDUCATION":
                key_m = "HI_EDU"
            elif key == "LINKEDIN_URL":
                key_m = "LN_URL"
            elif key == "LinkedIn URL":
                key_m = "Ln_URL"
            elif key == "PASSOUTYEAR":
                key_m = "PASS"
            elif key == "PHONE_NO":
                key_m = "PH_NO"
            elif key == "TOTAL_EXP":
                key_m = "TOT_EXP"
            elif key == "WORKING_DATES":
                key_m = "W_D"
            elif key == "PERSON_ADDRESS":
                key_m = "P_A"
            else:
                key_m = key
            new_count_dict_1[key_m] = round((count_dict_1[key] / sum_of_entities) * 100, 2)
        entities_1 = []
        values_1 = []
        for k in sorted(new_count_dict_1):
            entities_1.append(k)
            values_1.append(new_count_dict_1[k])
        print("new_count_dict_1", new_count_dict_1)
        count_dict_2 = {}
        for d in data_2:
            temp_entities = d[1]['entities']
            for ent in temp_entities:
                if ent[2] in count_dict_2.keys():
                    count_dict_2[ent[2]] = count_dict_2[ent[2]] + 1
                else:
                    count_dict_2[ent[2]] = 1
        sum_of_entities = sum(count_dict_2.values())
        new_count_dict_2 = {}
        for key in count_dict_2.keys():
            if key == "AWARDS":
                key_m = "AW"
            elif key == "CERTIFICATIONS":
                key_m = "CT"
            elif key == "CLIENT":
                key_m = "CL"
            elif key == "COMPANY":
                key_m = "COM"
            elif key == "COMPANY_LOC":
                key_m = "COM_LOC"
            elif key == "DESGINATION":
                key_m = "DESG"
            elif key == "EDUCATIONAL_INSTITUTION":
                key_m = "E_I"
            elif key == "GITHUB_URL":
                key_m = "G_U"
            elif key == "HIGHER_EDUCATION":
                key_m = "HI_EDU"
            elif key == "LINKEDIN_URL":
                key_m = "LN_URL"
            elif key == "LinkedIn URL":
                key_m = "Ln_URL"
            elif key == "PASSOUTYEAR":
                key_m = "PASS"
            elif key == "PHONE_NO":
                key_m = "PH_NO"
            elif key == "TOTAL_EXP":
                key_m = "TOT_EXP"
            elif key == "WORKING_DATES":
                key_m = "W_D"
            elif key == "PERSON_ADDRESS":
                key_m = "P_A"
            else:
                key_m = key
            new_count_dict_2[key_m] = round((count_dict_2[key] / sum_of_entities) * 100, 2)
        entities_2 = []
        values_2 = []
        for k in sorted(new_count_dict_2):
            entities_2.append(k)
            values_2.append(new_count_dict_2[k])
        print("new_count_dict_2", new_count_dict_2)
        import matplotlib.pyplot as plt
        c1, c2 = st.columns((10, 10))
        with c1:
            fig_9 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[0:5], values_1[0:5], alpha=0.75, label="few months back data",
                    width=0.4)
            plt.bar(entities_2[0:5], values_2[0:5], alpha=0.5, label="present data",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_9)
            fig_10 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[5:10], values_1[5:10], alpha=0.75, label="few months back data",
                    width=0.4)
            plt.bar(entities_2[5:10], values_2[5:10], alpha=0.5, label="present data",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_10)
        with c2:
            fig_11 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[10:15], values_1[10:15], alpha=0.75, label="few months back data",
                    width=0.4)
            plt.bar(entities_2[10:15], values_2[10:15], alpha=0.5, label="present data",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_11)
            fig_12 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[15:], values_1[15:], alpha=0.75, label="few months back data",
                    width=0.4)
            plt.bar(entities_2[15:], values_2[15:], alpha=0.5, label="present data",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_12)
    elif button_5:
        f_1 = open('utilities/dataset_0.7436912184166055.json')
        f_2 = open('utilities/validation.json')
        data_1 = json.load(f_1)
        data_2 = json.load(f_2)
        count_dict_1 = {}
        for d in data_1:
            temp_entities = d[1]['entities']
            for ent in temp_entities:
                if ent[2] in count_dict_1.keys():
                    count_dict_1[ent[2]] = count_dict_1[ent[2]] + 1
                else:
                    count_dict_1[ent[2]] = 1
        sum_of_entities = sum(count_dict_1.values())
        new_count_dict_1 = {}
        for key in count_dict_1.keys():
            if key == "AWARDS":
                key_m = "AW"
            elif key == "CERTIFICATIONS":
                key_m = "CT"
            elif key == "CLIENT":
                key_m = "CL"
            elif key == "COMPANY":
                key_m = "COM"
            elif key == "COMPANY_LOC":
                key_m = "COM_LOC"
            elif key == "DESGINATION":
                key_m = "DESG"
            elif key == "EDUCATIONAL_INSTITUTION":
                key_m = "E_I"
            elif key == "GITHUB_URL":
                key_m = "G_U"
            elif key == "HIGHER_EDUCATION":
                key_m = "HI_EDU"
            elif key == "LINKEDIN_URL":
                key_m = "LN_URL"
            elif key == "LinkedIn URL":
                key_m = "Ln_URL"
            elif key == "PASSOUTYEAR":
                key_m = "PASS"
            elif key == "PHONE_NO":
                key_m = "PH_NO"
            elif key == "TOTAL_EXP":
                key_m = "TOT_EXP"
            elif key == "WORKING_DATES":
                key_m = "W_D"
            elif key == "PERSON_ADDRESS":
                key_m = "P_A"
            else:
                key_m = key
            new_count_dict_1[key_m] = round((count_dict_1[key] / sum_of_entities) * 100, 2)
        entities_1 = []
        values_1 = []
        for k in sorted(new_count_dict_1):
            entities_1.append(k)
            values_1.append(new_count_dict_1[k])

        count_dict_2 = {}
        for d in data_2:
            temp_entities = d[1]['entities']
            for ent in temp_entities:
                if ent[2] in count_dict_2.keys():
                    count_dict_2[ent[2]] = count_dict_2[ent[2]] + 1
                else:
                    count_dict_2[ent[2]] = 1
        sum_of_entities = sum(count_dict_2.values())
        new_count_dict_2 = {}
        for key in count_dict_2.keys():
            if key == "AWARDS":
                key_m = "AW"
            elif key == "CERTIFICATIONS":
                key_m = "CT"
            elif key == "CLIENT":
                key_m = "CL"
            elif key == "COMPANY":
                key_m = "COM"
            elif key == "COMPANY_LOC":
                key_m = "COM_LOC"
            elif key == "DESGINATION":
                key_m = "DESG"
            elif key == "EDUCATIONAL_INSTITUTION":
                key_m = "E_I"
            elif key == "GITHUB_URL":
                key_m = "G_U"
            elif key == "HIGHER_EDUCATION":
                key_m = "HI_EDU"
            elif key == "LINKEDIN_URL":
                key_m = "LN_URL"
            elif key == "LinkedIn URL":
                key_m = "Ln_URL"
            elif key == "PASSOUTYEAR":
                key_m = "PASS"
            elif key == "PHONE_NO":
                key_m = "PH_NO"
            elif key == "TOTAL_EXP":
                key_m = "TOT_EXP"
            elif key == "WORKING_DATES":
                key_m = "W_D"
            elif key == "PERSON_ADDRESS":
                key_m = "P_A"
            else:
                key_m = key
            new_count_dict_2[key_m] = round((count_dict_2[key] / sum_of_entities) * 100, 2)
        entities_2 = []
        values_2 = []
        new_count_dict_2_m = {}
        for k in new_count_dict_1.keys():
            if k not in new_count_dict_2.keys():
                new_count_dict_2_m[k] = 0.1
            else:
                new_count_dict_2_m[k] = new_count_dict_2[k]

        for k in sorted(new_count_dict_2_m):
            entities_2.append(k)
            values_2.append(new_count_dict_2_m[k])

        import matplotlib.pyplot as plt

        c1, c2 = st.columns((10, 10))
        with c1:
            fig_9 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[0:5], values_1[0:5], alpha=0.75, label="reference",
                    width=0.4)
            plt.bar(entities_2[0:5], values_2[0:5], alpha=0.5, label="test",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_9)
            fig_10 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[5:10], values_1[5:10], alpha=0.75, label="reference",
                    width=0.4)
            plt.bar(entities_2[5:10], values_2[5:10], alpha=0.5, label="test",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_10)
        with c2:
            # fig_11 = plt.figure(figsize=(10, 10))
            # plt.bar(entities_1[10:15], values_1[10:15], alpha=0.75, label="reference",
            #         width=0.4)
            # plt.bar(entities_2[10:15], values_2[10:15], alpha=0.5, label="test",
            #         width=0.4)
            # plt.legend()
            # st.pyplot(fig_11)
            fig_12 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[15:], values_1[15:], alpha=0.75, label="reference",
                    width=0.4)
            plt.bar(entities_2[15:], values_2[15:], alpha=0.5, label="test",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_12)
            fig_11 = plt.figure(figsize=(10, 10))
            plt.bar(entities_1[10:15], values_1[10:15], alpha=0.75, label="reference",
                    width=0.4)
            plt.bar(entities_2[10:15], values_2[10:15], alpha=0.5, label="test",
                    width=0.4)
            plt.legend()
            st.pyplot(fig_11)
            
    elif button_6:
        c1, c2 = st.columns((10, 10))

        def compute_lengths(data):
            length_dict = {}

            for d in data:
                temp_entities = d[1]['entities']
                for ent in temp_entities:
                    entity_type = ent[2]
                    entity_value = d[0][ent[0]:ent[1]]

                    if entity_type in length_dict:
                        length_dict[entity_type].append(len(entity_value))
                    else:
                        length_dict[entity_type] = [len(entity_value)]

            return length_dict

        def plot_histogram(column, title_text, data1, data2, label1, label2):
            with column: 
                st.write(title_text)
                lengths_1 = compute_lengths(data1)
                lengths_2 = compute_lengths(data2)
                
                common_keys = set(lengths_1.keys()).intersection(set(lengths_2.keys()))
                
                if len(common_keys) == 1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(lengths_1[list(common_keys)[0]], bins=20, alpha=0.5, label=label1)
                    ax.hist(lengths_2[list(common_keys)[0]], bins=20, alpha=0.5, label=label2)
                    ax.set_title(list(common_keys)[0])
                    ax.set_xlabel('Length')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                else:
                    fig, axs = plt.subplots(len(common_keys), 1, figsize=(10, 5 * len(common_keys)))
                    for ax, entity in zip(axs, common_keys):
                        ax.hist(lengths_1[entity], bins=20, alpha=0.5, label=label1)
                        ax.hist(lengths_2[entity], bins=20, alpha=0.5, label=label2)
                        ax.set_title(entity)
                        ax.set_xlabel('Length')
                        ax.set_ylabel('Frequency')
                        ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

        with open('utilities/dataset_0.6954225242382301.json') as f_1:
            data_yesterday = json.load(f_1)
            
        with open('utilities/dataset_0.7436912184166055.json') as f_2:
            data_today = json.load(f_2)
            
        with open('utilities/validation.json') as f_3:
            data_validation = json.load(f_3)
            
        plot_histogram(c1, "Yesterday vs Today comparison", data_yesterday, data_today, 'f1 (yesterday)', 'f2 (today)')
        plot_histogram(c2, "Train vs Validation comparison", data_today, data_validation, 'Train', 'Validation')
     
    else:
        home()
        
    