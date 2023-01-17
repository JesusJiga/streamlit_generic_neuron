import streamlit as st
import neuron
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="Neuron!",
    menu_items={
        'Get Help': 'https://jesusjiga.es',
        'About': "# This is a classroom exercise"
    }
)

st.title("Neuron!")
st.image('neurona.png', width= 200)

num_neurons = st.slider("Seleccione el número de neuronas", 0, 10)

x = np.ones(num_neurons)
w = np.ones(num_neurons)

if(num_neurons > 0):
    columns = st.columns(num_neurons)
    for i in range(num_neurons):
        with columns[i]:
            w[i] = st.number_input(f'Selecciona el peso {i}', key="sliderCol"+str(i))
            x[i] = st.number_input(f'Elija la entrada {i}', key="numberCol"+str(i))
    col1, col2 = st.columns(2)
    with col1:
        b = st.number_input('Seleccione el sesgo')
    with col2:
        fActivacion = st.selectbox("Introduzca la función de activación", [neuron.Neuron.RELU, neuron.Neuron.TANH, neuron.Neuron.SIGMOID])
    
    if st.button("Calcular"):
        n1 = neuron.Neuron(w, b, fActivacion)
        st.write("La salida es: ", n1.run(input_data=x))