#Importando as bibliotecas
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from PIL import Image


#Título
st.subheader("Predição de Diabetes Utilizando Machine Learning")

#Imagem
#image = Image.open('C:/Users/Filipe/Desktop/ML/DiabetesApp.jpg')
#st.image(image, caption="ML", use_column_width= True)

st.write("""
\n
Aplicativo que faz a predição de possível diabetes dos
pacientes, através de inteligência artificial,
utilizando exames simples.  O menu na lateral servirá para 
o usuário inserir suas informações.
\n
Autores: José Filipe de Andrade e Maycon Carvalho.\n
Classificador Utilizado: Árvore de Decisão.\n
Base de Dados: PIMA - INDIA (kaggle)
""")

#dataset
df = pd.read_csv("C:/Users/Filipe/Desktop/ML/diabetes.csv")

#Cabeçalho
st.subheader("Visão das Informações")



#Nome do Usuário
user_input = st.sidebar.text_input("Digite seu nome")

st.write('Paciente: ', user_input)

#Dados de Entrada
X = df.drop(["Outcome"],1)
Y = df["Outcome"]

#Separa dados em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42 )

#Dados dos usuários com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez',0,15,1)
    glucose = st.sidebar.slider('Glicose',0,200,110)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea',0,122,72)
    skin_thickness = st.sidebar.slider('Espessura da Pele',0,99,20)
    insulin = st.sidebar.slider("Insulina",0,900,30)
    bmi = st.sidebar.slider("Índice de Massa Corporal",0.0,70.0,15.0)
    dpf = st.sidebar.slider("Histórico Familiar de Diabetes", 0.0,3.0,0.0)
    age = st.sidebar.slider('Idade', 16,100,21)

    user_data = { 'Gavidez': pregnancies,
                  'Glicose': glucose,
                  'Pressão Sanguínea': blood_pressure,
                  'Espessura da Pele': skin_thickness,
                  'Insulina': insulin,
                  'Índice de Massa Corporal': bmi,
                  'Histórico Familiar de Diabetes': dpf,
                  'Idade': age
                 }
    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_data()

#Gráfico
st.write("Comparando os dados:")
graf = st.bar_chart(user_input_variables)

st.subheader('Dados do Usuário')
st.write(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy',max_depth=3)
dtc.fit(X_train, Y_train)

#Acurácia do modelo
st.subheader('Acurácia do Modelo')
st.write(accuracy_score(Y_test, dtc.predict(X_test))*100)

#Previsão
prediction = dtc.predict(user_input_variables)

st.subheader("Previsão: ")
st.write("Se resultar em 1, a predição é que seria um caso de diabetes para os dados informados.")
st.write(prediction)

user_result = prediction
st.subheader("Seu Resultado:")
output = ''
if user_result[0]==0:
    output = 'Você está saudável'
else:
    output = 'Você não está saudável'
st.write(output)
