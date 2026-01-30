import streamlit as st
import pandas as pd
import numpy as np
from final_framework_streamlit import sloth_LCS_based, sloth_mapping_based
from utils import parse_table_from_dataframe, estimate_mapping_density
from testing_tables.test_tables import table_stocks_a, table_stocks_b

st.set_page_config(page_title="Framework Massima Sovrapposizione tra due Tabelle", layout="wide")

st.title("📊 Framework Massima Sovrapposizione tra Tabelle")
st.markdown(
    "Carica le tue tabelle o scegli un caso di test predefinito per calcolare la massima sovrapposizione tra due tabelle. "
    "In altre parole, trova la più grande sottotabella rettangolare comune tra le due tabelle fornite."
)

# === Funzione di reset ===
def reset_app():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# === Tabelle di test predefinite ===
r_tab = pd.read_csv('testing_tables/reverse_rows_cols#1.csv', sep=';')
r_tab_col_list1 = parse_table_from_dataframe(r_tab)
s_tab = pd.read_csv('testing_tables/reverse_rows_cols#2.csv', sep=';')
s_tab_col_list1 = parse_table_from_dataframe(s_tab)

r_tab = pd.read_csv('testing_tables/reverse_rows#1.csv', sep=';')
r_tab_col_list2 = parse_table_from_dataframe(r_tab)
s_tab = pd.read_csv('testing_tables/reverse_rows#2.csv', sep=';')
s_tab_col_list2 = parse_table_from_dataframe(s_tab)

test_sets = {
    "Test 1": {"table_a": table_stocks_a, "table_b": table_stocks_b},
    "Test 2": {"table_a": r_tab_col_list1, "table_b": table_stocks_b},
    "Test 3": {"table_a": r_tab_col_list1, "table_b": s_tab_col_list1}
}

test_cases = {
    "Test 1": "Le colonne 1 e 2 sono invertite e in entrambe le tabelle ci sono righe che non compaiono nell’altra.",
    "Test 2": "Le due tabelle non condividono alcun valore.",
    "Test 3": "Tutte le colonne e le righe sono invertite nell’ordine."
}

# === Sezione caricamento file ===
st.markdown("## 📤 Carica le tue tabelle")

uploaded_a = st.file_uploader("Carica la Tabella A (CSV)", type=["csv"], key="upload_a")
uploaded_b = st.file_uploader("Carica la Tabella B (CSV)", type=["csv"], key="upload_b")

user_uploaded = False
table_a, table_b = None, None

if uploaded_a is not None and uploaded_b is not None:
    try:
        table_a = pd.read_csv(uploaded_a, sep=None, engine='python')
        table_b = pd.read_csv(uploaded_b, sep=None, engine='python')
        table_a = parse_table_from_dataframe(table_a)
        table_b = parse_table_from_dataframe(table_b)
        user_uploaded = True
        st.success("✅ Tabelle personalizzate caricate correttamente.")
    except Exception as e:
        st.error(f"Errore durante il caricamento dei file: {e}")

# === Se non sono state caricate tabelle, mostra i test predefiniti ===
if not user_uploaded:
    st.markdown("## 📌 Oppure scegli un caso di test predefinito")
    test_case = st.selectbox("📁 Seleziona un caso di test", ["Nessuno"] + list(test_sets.keys()), key="test_case")

    if test_case != "Nessuno":
        selected = test_sets[test_case]
        table_a = selected["table_a"]
        table_b = selected["table_b"]
        st.markdown(f"**Descrizione:** {test_cases[test_case]}")

# === Mostra le tabelle SOLO se presenti ===
if table_a is not None and table_b is not None:
    st.subheader("🔍 Tabelle di input")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tabella A**")
        st.dataframe(np.transpose(table_a))
    with c2:
        st.markdown("**Tabella B**")
        st.dataframe(np.transpose(table_b))

    # === Selezione vincoli ===
    st.subheader("⚙️ Configura i Vincoli di Sovrapposizione")
    options = {"nessuno": "none", "ordinamento": "ordered", "contiguità": "contiguous"}

    col1, col2 = st.columns(2)
    with col1:
        row_constraint = st.selectbox("🔗 Vincolo sulle righe", options.keys(), index=0, key="row_constraint")
    with col2:
        col_constraint = st.selectbox("🔗 Vincolo sulle colonne", options.keys(), index=0, key="col_constraint")

    threshold_dup = 0.1
    flag = ''

    # === Calcolo sovrapposizione ===
    if st.button("Calcola Sovrapposizione", key="compute"):
        try:
            mapping_density_metrics = estimate_mapping_density(r_tab, s_tab, threshold_dup=0.2)
            R = mapping_density_metrics['Mapping_density_ratio']
            if R < threshold_dup:
                flag = 'Mapping'
                results, metrics = sloth_mapping_based(
                    table_a, table_b, verbose=False, metrics=[],
                    row_constraint=row_constraint, col_constraint=col_constraint
                )
            else:
                results, metrics = sloth_LCS_based(
                    table_a, table_b, verbose=False, metrics=[],
                    row_constraint=row_constraint, col_constraint=col_constraint
                )
                flag = 'LCS'

            st.markdown("---")
            st.subheader("✅ Risultato della Sovrapposizione")
            st.markdown(
                f"**Configurazione dei vincoli selezionata:** righe = `{row_constraint}`, colonne = `{col_constraint}`"
            )

            if metrics[0] > 1:
                st.markdown(
                    f"**Dimensioni della massima sovrapposizione trovata:** `{metrics[10]}` colonne, `{metrics[11]}` righe, `{metrics[12]}` celle"
                )

                st.markdown("### 📈 Tabella Massima di Sovrapposizione")
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    st.dataframe(results[1], width='content')

                st.markdown("### 🧮 Metriche aggiuntive")
                st.markdown(f"**Tempo di esecuzione:** {metrics[13]} secondi")
                st.markdown(f"**Numero di seed trovati:** {metrics[0]}")
                st.markdown(f"**Metodo utilizzato:** {flag}-based")

            else:
                st.subheader("‼️ Nessuna sovrapposizione trovata")

            # === Pulsante per tornare alla schermata iniziale ===
            st.markdown("---")
            if st.button("🔄 Ricomincia", key="reset"):
                reset_app()

        except ValueError as e:
            st.error(f"Errore: {e}")
else:
    st.info("👆 Carica due tabelle oppure seleziona un caso di test per visualizzarle.")
