```python
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Настройка страницы
# -----------------------------
st.set_page_config(
    page_title="Адаптация животных к городской среде",
    layout="wide"
)

# -----------------------------
# Загрузка данных
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("urban_wildlife_russian.csv")

df = load_data()

# -----------------------------
# Заголовок
# -----------------------------
st.title("🐾 Анализ адаптации животных к городской среде")

st.markdown("""
Дашборд предназначен для анализа поведения животных в городской среде,
оценки влияния шума, плотности населения и типа локации на их активность.
""")

# -----------------------------
# Боковая панель
# -----------------------------
st.sidebar.header("Фильтры")

species = st.sidebar.multiselect(
    "Вид животного",
    options=df["Вид_животного"].unique(),
    default=df["Вид_животного"].unique()
)

location = st.sidebar.multiselect(
    "Тип локации",
    options=df["Тип_локации"].unique(),
    default=df["Тип_локации"].unique()
)

time_obs = st.sidebar.multiselect(
    "Время наблюдения",
    options=df["Время_наблюдения"].unique(),
    default=df["Время_наблюдения"].unique()
)

adaptation = st.sidebar.multiselect(
    "Сигнал адаптации",
    options=df["Сигнал_адаптации"].unique(),
    default=df["Сигнал_адаптации"].unique()
)

filtered_df = df[
    (df["Вид_животного"].isin(species)) &
    (df["Тип_локации"].isin(location)) &
    (df["Время_наблюдения"].isin(time_obs)) &
    (df["Сигнал_адаптации"].isin(adaptation))
]

# -----------------------------
# KPI
# -----------------------------
st.subheader("Ключевые показатели")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Количество наблюдений",
    len(filtered_df)
)

col2.metric(
    "Средняя дистанция (км)",
    round(
        filtered_df["Дневная_дистанция_км"].mean(),
        2
    )
)

col3.metric(
    "Средний уровень шума",
    round(
        filtered_df["Уровень_шума_дБ"].mean(),
        1
    )
)

col4.metric(
    "Средняя плотность людей",
    round(
        filtered_df["Плотность_людей"].mean(),
        1
    )
)

st.divider()

# -----------------------------
# График 1
# -----------------------------
st.subheader("Типы адаптации животных")

adapt_counts = (
    filtered_df["Сигнал_адаптации"]
    .value_counts()
    .reset_index()
)

adapt_counts.columns = [
    "Сигнал адаптации",
    "Количество"
]

fig1 = px.bar(
    adapt_counts,
    x="Сигнал адаптации",
    y="Количество"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# График 2
# -----------------------------
st.subheader("Средняя дневная дистанция по видам животных")

species_avg = (
    filtered_df
    .groupby("Вид_животного")[
        "Дневная_дистанция_км"
    ]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    species_avg,
    x="Дневная_дистанция_км",
    y="Вид_животного",
    orientation="h"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# График 3
# -----------------------------
st.subheader("Влияние уровня шума на дистанцию")

fig3 = px.scatter(
    filtered_df,
    x="Уровень_шума_дБ",
    y="Дневная_дистанция_км",
    color="Вид_животного"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# График 4
# -----------------------------
st.subheader("Средняя дистанция по типам локации")

location_avg = (
    filtered_df
    .groupby("Тип_локации")[
        "Дневная_дистанция_км"
    ]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    location_avg,
    x="Тип_локации",
    y="Дневная_дистанция_км"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# График 5
# -----------------------------
st.subheader(
    "Тепловая карта активности животных"
)

heatmap = (
    filtered_df
    .pivot_table(
        values="Дневная_дистанция_км",
        index="Вид_животного",
        columns="Тип_локации",
        aggfunc="mean"
    )
)

fig5 = px.imshow(
    heatmap,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# Таблица
# -----------------------------
st.subheader("Данные")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# -----------------------------
# Выводы
# -----------------------------
st.subheader("Выводы")

st.markdown("""
- Дашборд позволяет анализировать адаптацию животных к городской среде.
- Можно изучать влияние шума и плотности населения на активность животных.
- Фильтры помогают исследовать отдельные виды животных и типы локаций.
- Тепловая карта показывает различия поведения животных в разных условиях городской среды.
""")
```

