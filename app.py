import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================

st.set_page_config(
    page_title="Адаптация животных к городской среде",
    page_icon="🐾",
    layout="wide"
)

# =====================================================
# ЗАГРУЗКА ДАННЫХ
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "urban_wildlife_adaptation_russian.csv"
    )

df = load_data()

# =====================================================
# ЗАГОЛОВОК
# =====================================================

st.title("🐾 Анализ адаптации животных к городской среде")

st.markdown("""
Данный проект посвящен исследованию поведения животных в городской среде.
В работе выполнен анализ данных и построена модель машинного обучения
для прогнозирования суточной дистанции перемещения животных.
""")

# =====================================================
# БОКОВАЯ ПАНЕЛЬ
# =====================================================

st.sidebar.header("Фильтры")

species_filter = st.sidebar.multiselect(
    "Вид",
    options=sorted(df["Вид"].unique()),
    default=sorted(df["Вид"].unique())
)

location_filter = st.sidebar.multiselect(
    "Тип локации",
    options=sorted(df["Тип_локации"].unique()),
    default=sorted(df["Тип_локации"].unique())
)

time_filter = st.sidebar.multiselect(
    "Время наблюдения",
    options=sorted(df["Время_наблюдения"].unique()),
    default=sorted(df["Время_наблюдения"].unique())
)

adapt_filter = st.sidebar.multiselect(
    "Сигнал адаптации",
    options=sorted(df["Сигнал_адаптации"].unique()),
    default=sorted(df["Сигнал_адаптации"].unique())
)

filtered_df = df[
    (df["Вид"].isin(species_filter))
    & (df["Тип_локации"].isin(location_filter))
    & (df["Время_наблюдения"].isin(time_filter))
    & (df["Сигнал_адаптации"].isin(adapt_filter))
]

# =====================================================
# KPI
# =====================================================

st.header("📊 Ключевые показатели")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Количество наблюдений",
    len(filtered_df)
)

col2.metric(
    "Средняя дистанция (км)",
    round(
        filtered_df[
            "Оценка_суточной_дистанции_км"
        ].mean(),
        2
    )
)

col3.metric(
    "Средний уровень шума",
    round(
        filtered_df[
            "Уровень_шума_дБ"
        ].mean(),
        2
    )
)

col4.metric(
    "Средняя плотность населения",
    round(
        filtered_df[
            "Плотность_населения"
        ].mean(),
        2
    )
)

st.divider()

# =====================================================
# ГРАФИК 1
# =====================================================

st.subheader("1. Распределение сигналов адаптации")

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

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =====================================================
# ГРАФИК 2
# =====================================================

st.subheader(
    "2. Средняя суточная дистанция по видам животных"
)

species_avg = (
    filtered_df
    .groupby("Вид")[
        "Оценка_суточной_дистанции_км"
    ]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    species_avg,
    x="Оценка_суточной_дистанции_км",
    y="Вид",
    orientation="h"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================================
# ГРАФИК 3
# =====================================================

st.subheader(
    "3. Влияние уровня шума на дистанцию"
)

fig3 = px.scatter(
    filtered_df,
    x="Уровень_шума_дБ",
    y="Оценка_суточной_дистанции_км",
    color="Вид"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================================
# ГРАФИК 4
# =====================================================

st.subheader(
    "4. Средняя дистанция по типам локации"
)

location_avg = (
    filtered_df
    .groupby("Тип_локации")[
        "Оценка_суточной_дистанции_км"
    ]
    .mean()
    .reset_index()
)

fig4 = px.bar(
    location_avg,
    x="Тип_локации",
    y="Оценка_суточной_дистанции_км"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =====================================================
# ГРАФИК 5
# =====================================================

st.subheader(
    "5. Тепловая карта активности животных"
)

heatmap = filtered_df.pivot_table(
    values="Оценка_суточной_дистанции_км",
    index="Вид",
    columns="Тип_локации",
    aggfunc="mean"
)

fig5 = px.imshow(
    heatmap,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

st.divider()

# =====================================================
# МАШИННОЕ ОБУЧЕНИЕ
# =====================================================

st.header("🤖 Машинное обучение")

st.write("""
Модель Random Forest Regressor прогнозирует
суточную дистанцию перемещения животного
на основе характеристик городской среды.
""")

target = "Оценка_суточной_дистанции_км"

X = df.drop(
    columns=[
        "ID_животного",
        target
    ]
)

y = df[target]

categorical_features = [
    "Вид",
    "Время_наблюдения",
    "Тип_локации",
    "Сигнал_адаптации"
]

numeric_features = [
    "Уровень_шума_дБ",
    "Плотность_населения",
    "Оценка_источника_пищи",
    "Оценка_качества_укрытия",
    "Оценка_аномалии_поведения"
]

preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numeric_features
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_features
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)

st.metric(
    "Качество модели (R²)",
    round(r2, 3)
)

# =====================================================
# ВАЖНОСТЬ ПРИЗНАКОВ
# =====================================================

st.subheader("Важность признаков")

rf_model = model.named_steps["regressor"]

ohe = (
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
)

feature_names = (
    numeric_features
    + list(
        ohe.get_feature_names_out(
            categorical_features
        )
    )
)

importance_df = pd.DataFrame({
    "Признак": feature_names,
    "Важность": rf_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Важность",
        ascending=False
    )
    .head(10)
)

fig_imp = px.bar(
    importance_df,
    x="Важность",
    y="Признак",
    orientation="h"
)

st.plotly_chart(
    fig_imp,
    use_container_width=True
)

# =====================================================
# ПРОГНОЗ
# =====================================================

st.header("🔮 Прогноз суточной дистанции")

col1, col2 = st.columns(2)

with col1:

    species = st.selectbox(
        "Вид животного",
        sorted(df["Вид"].unique())
    )

    observation_time = st.selectbox(
        "Время наблюдения",
        sorted(df["Время_наблюдения"].unique())
    )

    location = st.selectbox(
        "Тип локации",
        sorted(df["Тип_локации"].unique())
    )

    adaptation = st.selectbox(
        "Сигнал адаптации",
        sorted(df["Сигнал_адаптации"].unique())
    )

with col2:

    noise = st.slider(
        "Уровень шума",
        30.0,
        100.0,
        60.0
    )

    density = st.slider(
        "Плотность населения",
        0.0,
        100.0,
        50.0
    )

    food = st.slider(
        "Оценка источника пищи",
        1,
        10,
        5
    )

    shelter = st.slider(
        "Оценка качества укрытия",
        1,
        10,
        5
    )

    anomaly = st.slider(
        "Оценка аномалии поведения",
        0.0,
        1.0,
        0.5
    )

if st.button("Сделать прогноз"):

    sample = pd.DataFrame({
        "Вид": [species],
        "Время_наблюдения": [observation_time],
        "Тип_локации": [location],
        "Уровень_шума_дБ": [noise],
        "Плотность_населения": [density],
        "Оценка_источника_пищи": [food],
        "Оценка_качества_укрытия": [shelter],
        "Оценка_аномалии_поведения": [anomaly],
        "Сигнал_адаптации": [adaptation]
    })

    prediction = model.predict(sample)[0]

    st.success(
        f"Прогнозируемая суточная дистанция: {prediction:.2f} км"
    )
