import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Deployment purposes
import joblib
from custom_transformer import NewColumnTransform


# Sidebar widget
st.sidebar.header('Menu')
# loading our model
model = joblib.load("my_model.pkl")


def main():
    page = st.sidebar.selectbox(
        "Select a page", ["Dashboard", "Exploration","Model Accuracy", "Model Prediction"])

    if page == "Dashboard":
        homepage_screen()
    elif page == "Exploration":
        exploration_screen()
    elif page == "Model Accuracy":
        model_screen()
    elif page == "Model Prediction":
        model_predict()


@st.cache_data()
def load_data():
    data = pd.read_csv('clean_data.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():

    color = "#e44652"
    text = "Simple Data Dashboard"
    st.markdown(f"<h1 style='color:{color};'>{text}</h1>", unsafe_allow_html=True)

    # st.divider()
    st.markdown("The dataset used in this project is utilized to build a machine learning model that predicts a person's insomnia disease based on the provided health and condition information. The dataset consists of 70,000 samples from different individuals' health records. Each record contains 11 different pieces of information along with their insomnia report. More details about the dataset can be accessed [here]('https://www.kaggle.com/c/idao-2022-bootcamp-insomnia').")
    st.subheader('Column Description',divider=True)

    st.markdown("""
                ```
                1.  code         : person's identificator 
                2.  age          : person's age in years
                3.  weight       : person's weight in kilograms
                4.  height       : person's height in centimeters 
                5.  sex          : person's sex
                6.  stress       : level of stress during last month (1, 2, 3 - higher values correspond to larger stress)
                7.  doctor       : relative number of visits to doctor previously (1, 2, 3 - higher values correspond to greater number of visits)
                8.  sport        : is person physically active or not (binary)
                9.  pernicious_1 : does person have some bad habit or not (binary)
                10. pernicious_2 : does person have some another bad habit or not (binary)
                11. ubp/lbp      : upper/lower blood pressure in mmHg 
                12. insomnia     : target, does person have sleep disorder or not (binary)
                ```
                """)  
            #    
            #   
            #   
            #   
            #  
            # insomnia - target, does person have sleep disorder or not (binary)  
    
    if st.checkbox('Show Data Preview'):
        # Load data
        data_load_state = st.text('Loading data...')
        st.write(df.head())
        data_load_state.text('')
        # st.bar_chart(df)
        # st.line_chart(df)

    st.subheader('Data Summary',divider=True)
    data_summary = df.describe().T
    type = df.dtypes.reset_index()
    type= type.rename(columns={0:'type'})
    uniques= df.nunique().reset_index()
    uniques = uniques.rename(columns = {0:'count_unique'})
    data = data_summary.merge(type,left_index=True,right_on='index')
    data = pd.merge(data,uniques,on='index')
    data =data.rename(columns={'index':'column'}).set_index('column')
    st.write(data)
    

    
    st.subheader('Filter Data',divider=True)
  
    columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)

    selected_column = col1.selectbox('Select column to filter by', columns)
    unique_values = df[selected_column].unique()
    
    selected_value = col1.selectbox('Select value', unique_values)
    filtered_data = df[df[selected_column] == selected_value]
    filtered_data = filtered_data.reset_index(drop=True)
    st.write(filtered_data)
    
    count_value = filtered_data.shape[0]
    count_percentage =f"{round(count_value / (df.shape[0])*100,2)}%"
    col2.metric('Count Row',count_value,count_percentage,border=True)



def exploration_screen():
    color = "#e44652"
    text = "Data Exploration"
    st.markdown(f"<h1 style='color:{color};'>{text}</h1>", unsafe_allow_html=True)

    st.write(""" 
        This page contains general exploratory data analysis in order to get basic insight of the dataset information and get the feeling about what this dataset is about.
    """)

    st.subheader("Correlational Matrix",divider=True)
    st.write('This figure shows the correlation among features')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none") 
    corr_df = df[['age', 'sex', 'stress', 'doctor', 'ubp', 'lbp','insomnia']]
    corr = df.corr()
    mask = np.triu(corr)
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, mask=mask)
    st.write(fig)
    
    st.markdown("""
                From the figure above we got some insight related to the features which have high correlation such as :
        - blood pressure and insomnia = 0.31 for lbp and 0.42 for ubp
        - doctor and stress = 0.46
        - sex and height = 0.52
        - weight and height = 0.31
        """)
    
    st.subheader("Blood_pressure and Insomnia",divider=True)
    st.write("This scatterplot diagram shows how person's blood pressure: ubp (upper blood pressure) and lbp (lower blood pressure), bring the high contribution to give an insomnia disease.")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    sns.scatterplot(data=df, x='lbp', y='ubp', hue='insomnia', ax=ax,palette=['#e44652', '#e2c178'] )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)
    st.markdown("""
                Some points derived based on the diagram above:
                - lbp and ubp have positive correlation.
                - If the upper blood pressure increase, it will be followed by the increasing of lower blood pressure aswell.
                - The people with high blood pressure most likely will have insomnia problem. And the most influencing factor is the upper blood pressure because even the people with low lbp but high ubp they still tends to have insomia.
                """)


    st.subheader("Doctor and Stress Relation", divider=True)
    st.write("According to the correlation matrix showed above, there are high correlation between these two features: doctor and stress with 0.46.")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    sns.countplot(data=df, x='doctor', hue='stress', ax=ax, palette=['#F8894F','#AE1726','#006837'])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)
    st.markdown("""
                - Most people on the first and second level of stress visit doctor once only.
                - People who visit doctor 3 times are those whose stress level is at level 3 followed by the people with stress level at 1 and only few people with stress level 2 come to see doctor 3 times.
                - It can be concluded that people who has stress problem still rarely visit the doctor.
                """)
    
    
    st.subheader("Gender and Height Relation",divider=True)
    st.write("From this dataset, we can find that generally male is taller than female when the shorter male is 10 cm higher than the shortest female: the height of average female is around 165 cm when the male is around 170 cm. It seem's like the taller a person is the heavier he is.")
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    fig.patch.set_alpha(0)
    for ax in axs:
        ax.set_facecolor("none") 
    sns.scatterplot(data=df, x='height', y='weight', hue='sex', ax=axs[0],palette=['#2d1e3e','#e44652'])
    sns.kdeplot(data=df, x='height', hue='sex', ax=axs[1],palette=['#2d1e3e','#e44652'])
    plt.show()
    st.write(fig)
    
    st.divider()
    
    st.html("<h3>Now, let's count the total rows and have a look proportion of dataset based on column insomnia as the target value</h3>")


    # Create two columns
    col1, col2 = st.columns(2)  # col1 (wider for chart), col2 (narrower for text)

    # Create figure
    fig, axs = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0)  # Remove figure background
    axs.set_facecolor("none")  # Remove plot background

    # Bar chart with custom colors
    df['insomnia'].value_counts().plot(kind='bar', ax=axs, color=['#e44652', '#f06640'])

    # Display the chart in col1
    col1.pyplot(fig)

    # Create an empty container in col2 to center text vertically
    with col2:
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("The total row for the people who got insomnia and who did not is almost the same.")
        st.write("") 

    
    
    col1, col2 = st.columns(2)
    fig, axs = plt.subplots(figsize=(6,6))
    fig.patch.set_alpha(0)
    axs.set_facecolor("none") 
    df['insomnia'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=axs, colors=['#e44652', '#f06640'])
    with col1:
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("")  
        st.write("There are 51 % data of people with have no insomnia problems and 49 % people have insomnia. This is indicate that this dataset classified as fair dataset.")
        st.write("")  
    col2.write(fig)

    


def model_screen():
    st.subheader("Machine Learning Model")
    st.write("There are several machine learning models used to train the dataset in this project. We compare accuracy of each model and choose the highest accuracy among them to use in the prediction model.")
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
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    st.write(fig)
    st.write(f"""
    - Cross Validation  mean: {cross_score}   
    - Precision : {prec_score}  
    - Recall : {rec_score}  
    - F1 : {f1}  
    - Training Accuracy : {acc_score_train}  
    - Validation Accuracy : {acc_score_test}  
             """)


def model_predict():

    st.subheader("Prediction")
    st.html("<h4r>This is the final model to use for predicting person's insomnia disease based on the given features.</h4r>")
    st.html("<p>Field this form to predict the posibility of getting insomnia !</p>")
    # pernicious_1 = list(df.age.unique())
    # pernicious_2 = list(df.age.unique())
    # sport_value = list(df.education.unique())
    # doctor_value = list(df.default.unique())
    # stress_value = list(df.month.unique())
    col1,col2,col3 = st.columns(3)
    age = col1.number_input(label="Age", min_value=20,
                          max_value=100, step=1,)
    col1,col2,col3 = st.columns(3)
    weight = col1.slider(
        "Weight", 20, 150)
    height = col3.slider(
        "Height", 20, 250)
    stress = st.radio("Stress Level", ['1', '2', '3', '4'])
    doctor = st.radio("How many times consulting to a doctor", ['1', '2', '3'])
    sport = st.radio("Active in sport", ['yes', 'no'])
    col1,col3 = st.columns(2)
    pernicious_1 = col1.selectbox(
        "Do you have bad habit?", ['yes', 'no'])
    pernicious_2 = col3.selectbox(
        "Other bad habit?", ['yes', 'no'])
    col1,col2,col3 = st.columns(3)
    ubp = col1.slider("Upper blood pressure", 50, 200)
    lbp = col3.slider("Lower blood pressure", 50, 200)
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
