import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
# Machine Learning model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import TransformerMixin
# Deployment purposes
import joblib
from custom_transformer import NewColumnTransform


# Sidebar widget
st.sidebar.header('Menu')
# loading our model
model = joblib.load("my_model.pkl")


def main():
    page = st.sidebar.selectbox(
        "Select a page", ["Homepage", "Exploration", "Prediction"])

    if page == "Homepage":
        homepage_screen()
    elif page == "Exploration":
        exploration_screen()
    # elif page == "Model":
    #     model_screen()
    elif page == "Prediction":
        model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('clean_data.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():

    st.title('BANK TERM DEPOSIT CAMPAIGN')
    st.header("Dataset Information")
    st.write("""  
        **About Dataset**  

            id - person's identificator  
            age - person's age in years  
            weight- person's weight in kilograms  
            height- person's height in centimeters  
            sex - person's sex  
            stress - level of stress during last month (1, 2, 3 - higher values correspond to larger stress)  
            doctor - relative number of visits to doctor previously (1, 2, 3 - higher values correspond to greater number of visits)  
            sport - is person physically active or not (binary)  
            pernicious_1 - does person have some bad habit or not (binary)  
            pernicious_2 - does person have some another bad habit or not (binary)  
            ubp/lbp - upper/lower blood pressure in mmHg  
            insomnia - target, does person have sleep disorder or not (binary)    
        
        The dataset constains of 70000 instances and 13 columns. More information of the dataset can be accessed [here]("https://www.kaggle.com/c/idao-2022-bootcamp-insomnia").
        
    """)

    if st.checkbox('See dataset'):
        # Load data
        data_load_state = st.text('Loading data...')
        df = load_data()
        st.write(df)
        data_load_state.text('')


def exploration_screen():
    st.title("Data Exploration")
    st.write(""" 
        This page contains general exploratory data analysis in order to get basic insight of the dataset information and get the feeling about what this dataset is about.
    """)

    st.write("""
        ## 📌 Correlational Matrix  
        
    """)
    # Matrix correlation.
    fig, ax = plt.subplots()
    corr_df = df[['age', 'sex', 'stress', 'doctor', 'ubp', 'lbp',
                 'insomnia']]
    plt.figure(figsize=[10, 1])
    sns.heatmap(corr_df.corr(), annot=True, cmap='RdYlGn', ax=ax)
    # plt.style.available
    plt.title('Correlation Among Features')
    st.write(fig)

    st.write("""
        ## 📌  Features Correlation Value toward Target Column
        
    """)
    # Display correlation towards target column
    fig, axs = plt.subplots(figsize=(10, 4))
    corr = df.corr()['insomnia'].reset_index()
    # corr.drop( axis=0, inplace=True)
    sns.barplot(data=corr, x='index', y='insomnia', ax=axs)
    plt.xticks(rotation=70)
    st.write(fig)

    st.write("""
        ## 📌 Target Label Frequency  
        
    """)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    df['insomnia'].value_counts().plot(kind='bar', ax=axs[0])
    df['insomnia'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, ax=axs[1], colors=['green', 'teal'])
    st.write(fig)

    st.write("""
        ## 📌  Blood_pressure and Insomnia
        
    """)
    fig, axs = plt.subplots()
    sns.scatterplot(data=df, x='lbp', y='ubp', hue='insomnia', ax=axs)
    st.write(fig)

    st.write("""
        ## 📌 Gender and Height Relation.
        
    """)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    sns.scatterplot(data=df, x='height', y='weight', hue='sex', ax=axs[0])
    sns.kdeplot(data=df, x='height', hue='sex', ax=axs[1])
    plt.show()
    st.write(fig)

    st.write("""
        ## 📌  Doctor and Stress Relation
        
    """)
    fig, axs = plt.subplots()
    sns.countplot(data=df, x='doctor', hue='stress', ax=axs)
    st.write(fig)


def model_screen():
    st.title("Model")
    st.write(""" 
             
             """)
    model_selected = st.selectbox("Select model: ", [
                                  'Logistic Regression', 'SVC', 'KNeighbors Classifier', 'RandomForest Classifier', 'Multinomial NB', 'AdaBoost Classifier'])
    if model_selected == 'Logistic Regression':
        matrix = [[8778, 122], [875, 255]]
        cross_val = 0.8988667701399706
        precision = 0.6763925729442971
        recall = 0.22566371681415928
        F1 = 0.33842070338420704
        train_acc = 0.8990661038917876
        test_acc = 0.9005982053838485
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'SVC':
        matrix = [[8790, 110], [880, 250]]
        cross_val = 0.8978032607622813
        precision = 0.6944444444444444
        recall = 0.22123893805309736
        F1 = 0.33557046979865773
        train_acc = 0.8992655123134701
        test_acc = 0.901296111665005
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'RandomForest Classifier':
        matrix = [[8594, 306], [804, 326]]
        cross_val = 0.8886968091349463
        precision = 0.5158227848101266
        recall = 0.2884955752212389
        F1 = 0.3700340522133938
        train_acc = 0.9422712619229618
        test_acc = 0.8893320039880359
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'Multinomial NB':
        matrix = [[8534, 366], [750, 380]]
        cross_val = 0.88766667147192853
        precision = 0.5093833780160858
        recall = 0.336283185840708
        F1 = 0.4051172707889126
        train_acc = 0.8876998238558942
        test_acc = 0.8887337986041874
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'AdaBoost Classifier':
        matrix = [[8756,  144], [861, 269]]
        cross_val = 0.8986673738686439
        precision = 0.6513317191283293
        recall = 0.336283185840708
        F1 = 0.3486714193130266
        train_acc = 0.8989996344178935
        test_acc = 0.8998005982053838
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'KNeighbors Classifier':
        matrix = [[8761, 139], [915,  215]]
        cross_val = 0.8941805904650459
        precision = 0.6073446327683616
        recall = 0.1902654867256637
        F1 = 0.28975741239892183
        train_acc = 0.9026886902190169
        test_acc = 0.8949152542372881
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)


def validation(matrix, cross_score, prec_score, rec_score, f1, acc_score_train, acc_score_test):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    st.write(fig)
    st.write(f"""
    👉 Cross Validation  mean: {cross_score}   
    👉 Precision : {prec_score}  
    👉 Recall : {rec_score}  
    👉 F1 : {f1}  

    👉 Training Accuracy : {acc_score_train}  
    👉 Validation Accuracy : {acc_score_test}  
             """)


def model_predict():

    st.title("Prediction")
    st.write("### Field this form to predict the posibility of getting insomnia !")
    # pernicious_1 = list(df.age.unique())
    # pernicious_2 = list(df.age.unique())
    # sport_value = list(df.education.unique())
    # doctor_value = list(df.default.unique())
    # stress_value = list(df.month.unique())
    age = st.number_input(label="Age", min_value=30,
                          max_value=100, step=1,)
    weight = st.slider(
        "Weight", 20, 150)
    height = st.slider(
        "Height", 20, 250)
    stress = st.radio("Stress Level", ['1', '2', '3', '4'])
    doctor = st.radio("How many times visited doctor", ['1', '2', '3'])
    sport = st.radio("Active in sport", ['yes', 'no'])
    pernicious_1 = st.selectbox(
        "Do you have bad habit?", ['yes', 'no'])
    pernicious_2 = st.selectbox(
        "Do you have other bad habit?", ['yes', 'no'])
    ubp = st.slider("Upper blood pressure", 50, 200)
    lbp = st.slider("Lower blood pressure", 50, 200)
    submit_button = st.button("Predict")

    if sport == 'yes':
        sport = 1
    else:
        sport = 0

    if pernicious_1 == 'yes':
        pernicious_1 = 1
    else:
        pernicious_1 = 0

    if pernicious_2 == 'yes':
        pernicious_2 = 1
    else:
        pernicious_2 = 0
#
#     if pdays == 'yes':
#         pdays = 1
#     else:
#         pdays = 0
#     if has_loan == 'yes':
#         has_loan = 1
#     else:
#         has_loan = 0
#
#     if old_campaign == 'yes':
#         old_campaign = 1
#     else:
#         old_campaign = 0
#
    data = {
        'age': [age], 'weight': [weight], 'height': [height], 'stress': [stress], 'doctor': [doctor],
        'sport': [sport], 'pernicious_1': [pernicious_1], 'pernicious_2': [pernicious_2], 'ubp': [ubp], 'lbp': [lbp]
    }
#
    new_data = pd.DataFrame(data=data)

#
    if submit_button:

        result = model.predict(new_data)
        updated_res = result.flatten().astype(float)
        if updated_res[0] == 1:
            updated_res = "😱 Probably you have insomnia problem."
            # st.balloons
        else:
            updated_res = '🤩 Seems like you have no insomnia problem.'
        st.success(
            '{}'.format(updated_res))


main()
