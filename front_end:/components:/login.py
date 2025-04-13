import streamlit as st
from config import USUARIO_VALIDO, CONTRASENA_VALIDA

def login():
    st.title('Inicio de Sesión')

    usuario = st.text_input('Usuario', '')
    contrasena = st.text_input('Contraseña', '', type='password')

    if st.button('Iniciar sesión'):
        if usuario == USUARIO_VALIDO and contrasena == CONTRASENA_VALIDA:
            st.session_state.logged_in = True
            st.success('¡Inicio de sesión exitoso!')
        else:
            st.session_state.logged_in = False
            st.error('Usuario o contraseña incorrectos')
