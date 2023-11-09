import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Algorithm
from topology import Topology


df = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        {"command": "st.balloons", "rating": 5, "is_widget": False},
        {"command": "st.time_input", "rating": 3, "is_widget": True},
    ]
)

col1, col2 = st.columns(2)

a, b = 0, 0

with col1:
    a_input = st.text_input('Введите количество строк')

with col2:
    b_input = st.text_input('Введите количество колонок')

a = int(a_input) if len(a_input) > 0 else a
b = int(b_input) if len(b_input) > 0 else b

index = list(map(str, range(1,a + 1)))
columns = list(map(str, range(1,b + 1)))
df = pd.DataFrame(index=index, columns=columns)

edited_df = st.data_editor(df) # 👈 An editable dataframe
print(edited_df)

bt = st.button('Рандом')
if bt:
    df = Topology().generate_topology(
        'test_123.csv', 
        size=(a, b))
    
    edited_df = st.dataframe(df)

print(edited_df)

crossover, selection, mutate, mode = st.columns(4)

with crossover:
    cr_alg = st.radio(
        'Выберите алгоритм скрещивания',
        ['Одноточечное', 'Прямое']
    )

    if cr_alg == 'Одноточечное':
        crossover_algorithm = Algorithm().one_point_crossover
    elif cr_alg == 'Прямое':
        crossover_algorithm = Algorithm().direct_crossover

    print(crossover_algorithm)

with selection:
    sc_alg = st.radio(
        'Выберите алгоритм отбора особей',
        ['Рулетка', 'Турнир']
    )

    if sc_alg == 'Рулетка':
        selection_algorithm = Algorithm().roulette_selection
    else:
        selection_algorithm = Algorithm().tournament_selection

with mutate:
    mt_alg = st.radio(
        'Выберите алгоритм мутации особей',
        ['Обмен вершинами', 'Рандомная вершина']
    )

    if mt_alg == 'Обмен вершинами':
        mutate_algorithm = Algorithm().swap_mutate
    else:
        mutate_algorithm = Algorithm().random_mutate

with mode:
    mode_name = st.radio(
        'Выберите режим обучения',
        ['Пошаговый', 'Полный']
    )

    if mode_name == 'Пошаговый':
        mode = 'step'
    else:
        mode = 'full'

@st.cache_resource
def load_model():
    pass


