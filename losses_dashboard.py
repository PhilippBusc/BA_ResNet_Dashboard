import streamlit as st
import numpy as np
import joblib
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def sync_streamlit_theme_to_mpl():
    # read active Streamlit theme
    base   = st.get_option("theme.base")                     # "light" | "dark"
    primary= st.get_option("theme.primaryColor")
    bg     = st.get_option("theme.backgroundColor")
    sbg    = st.get_option("theme.secondaryBackgroundColor")
    text   = st.get_option("theme.textColor")
    font   = st.get_option("theme.font")                     # "sans serif" (default), "serif", "monospace"

    # map Streamlit font -> Matplotlib family
    family = {
        "sans serif": "sans-serif",
        "serif": "serif",
        "monospace": "monospace",
    }.get(font, "sans-serif")

    grid_color = "#D0D0D0" if base == "light" else "#3B3B3B"

    mpl.rcParams.update({
        # fonts
        "font.family": family,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # colors from theme
        "axes.edgecolor": text,
        "axes.labelcolor": text,
        "xtick.color": text,
        "ytick.color": text,
        "text.color": text,

        # grid/legend
        "grid.alpha": 0.35,
    })

sync_streamlit_theme_to_mpl()

st.set_page_config(page_title="ResNet-Approximation", page_icon="📈")
st.subheader("📈 ResNet-Approximation der Sinusfunktion")
st.info(
    "Dieses Dashboard dient als Ergänzung zur Bachelorarbeit "
    "**Konvergenzverhalten überparametrisierter residualer neuronaler Netzwerke**.\n"
    "Es visualisiert die empirischen Kosten im Verlauf des Trainings eines ResNets "
    "zur Approximation der Sinusfunktion für verschiedene Netzwerkarchitekturen "
    "(Breite und Tiefe), Dimensionen der Eingabedaten und Schrittweiten des Gradientenverfahrens."
)

# --- Daten laden ---
main_path = "Data/"
# all_losses_array = joblib.load(main_path + "all_losses_list.joblib")
all_losses_array = joblib.load(main_path + "all_losses_10_steps.joblib")
all_losses_array = np.array(all_losses_array)

# --- Parameteroptionen ---
h_values = [0.2, 0.4, 0.6, 0.8]
stop_values = [1e-10]
M_values   = [1, 2, 4, 8, 20, 50, 100]
L_values   = [1, 2, 4, 8, 20, 50, 100]
d_values   = [2, 3, 5, 11]
max_num_epochs = [100000]

# Alle Parameterkombinationen
param_list = list(itertools.product(h_values, M_values, L_values, stop_values, d_values, max_num_epochs))

d_values_input = [2, 3, 5, 11]

# Default-Input
h = h_values[2]
M = M_values[1]
L = L_values[1]
d = d_values_input[3]
s = 1000

epst = stop_values[0]
epo  = max_num_epochs[0]
selected_tuple = (h, M, L, epst, d, epo)

h_idx = h_values.index(h)
M_idx = M_values.index(M)
L_idx = L_values.index(L)
d_idx = d_values_input.index(d)

# --- Session State ---
if "M_idx" not in st.session_state:
    st.session_state.M_idx = []
if "L_idx" not in st.session_state:
    st.session_state.L_idx = []
if "h_idx" not in st.session_state:
    st.session_state.h_idx = []
if "d_idx" not in st.session_state:
    st.session_state.d_idx = []

st.session_state.M_idx.append(M_idx)
st.session_state.L_idx.append(L_idx)
st.session_state.h_idx.append(h_idx)
st.session_state.d_idx.append(d_idx)

if "extra_lines" not in st.session_state:
    st.session_state.extra_lines = []  # each item: {"label": str, "spec": tuple}

# Default-Linie nur einmal hinzufügen
if "add_state" not in st.session_state:
    st.session_state.add_state = True
    label = f"h = {h}, M = {M}, L = {L}, d = {d}"
    st.session_state.extra_lines.append({"label": label, "spec": selected_tuple})

# --- Modelle hinzufügen ---
expand_label = "➕ Modelle hinzufügen"
with st.expander(expand_label, expanded=False):
    if st.session_state.pop("_show_added", False):
        added = st.session_state.pop("_added_count", 0)
        if added:
            st.toast(f"{added} Modell(e) hinzugefügt.", icon="✅")
        else:
            st.toast("Keine neuen Modelle (evtl. Duplikate).", icon="ℹ️")

    col6, col5 = st.columns(2)
    col3, col2, col1, col4 = st.columns(4)

    default_h = [h_values[st.session_state.h_idx[-1]]] if st.session_state.get("h_idx") else [h_values[2]]
    default_M = [M_values[st.session_state.M_idx[-1]]] if st.session_state.get("M_idx") else [M_values[1]]
    default_L = [L_values[st.session_state.L_idx[-1]]] if st.session_state.get("L_idx") else [L_values[1]]
    default_d = [d_values_input[st.session_state.d_idx[-1]]] if st.session_state.get("d_idx") else [d_values_input[3]]

    st.session_state.setdefault("add_h", default_h)
    st.session_state.setdefault("add_M", default_M)
    st.session_state.setdefault("add_L", default_L)
    st.session_state.setdefault("add_d", default_d)

    # Trigger für "Alle auswählen"
    for trigger_key, target_key, all_vals in [
        ("_all_h_trigger", "add_h", h_values),
        ("_all_M_trigger", "add_M", M_values),
        ("_all_L_trigger", "add_L", L_values),
        ("_all_d_trigger", "add_d", d_values_input),
    ]:
        if st.session_state.get(trigger_key):
            st.session_state[target_key] = all_vals
            st.session_state[trigger_key] = False  # clear trigger

    with col1:
        add_h = st.multiselect(
            "Schrittweite (h)",
            options=h_values,
            key="add_h",
            help="Schrittweite des Gradientenverfahrens",
        )
        if set(add_h) != set(h_values):
            if st.button("Alle auswählen", key="all_h_btn"):
                st.session_state["_all_h_trigger"] = True
                st.rerun()

    with col2:
        add_M = st.multiselect(
            "Breite (M)",
            options=M_values,
            key="add_M",
            help="Breite des ResNet - Anzahl der Neuronen pro Schicht",
        )
        if set(add_M) != set(M_values):
            if st.button("Alle auswählen", key="all_M_btn"):
                st.session_state["_all_M_trigger"] = True
                st.rerun()

    with col3:
        add_L = st.multiselect(
            "Tiefe (L)",
            options=L_values,
            key="add_L",
            help="Tiefe des ResNet - Anzahl der Schichten",
        )
        if set(add_L) != set(L_values):
            if st.button("Alle auswählen", key="all_L_btn"):
                st.session_state["_all_L_trigger"] = True
                st.rerun()

    with col4:
        add_d = st.multiselect(
            "Dimension (d)",
            options=d_values_input,
            key="add_d",
            help="Dimension der Trainingsdaten",
        )
        if set(add_d) != set(d_values):
            if st.button("Alle auswählen", key="all_d_btn"):
                st.session_state["_all_d_trigger"] = True
                st.rerun()

    # s
    with col5:
        s = st.number_input(
            "Finale Iteration",
            value=1000,
            min_value=2,
            max_value=max_num_epochs[0],
            help=(
                "Die Iteration, bis zu der die Kosten im Plot angezeigt werden.\n"
                "Wenn Sie die Iteration ändern und mit **ENTER** bestätigen, wird der Plot automatisch angepasst. "
                "Die Modelle müssen nicht erneut hinzugefügt werden."
            ),
        )
        if s > 1000 and s % 10 != 0:
            s_adjusted = s - (s % 10)
            st.warning(
                f"Warnung: Die Anzahl der Iterationen ist größer als 1000 und nicht durch 10 teilbar. "
                f"Die Anzahl der Iterationen wird automatisch auf {s_adjusted} gesetzt. "
                "Die Kosten werden nicht für jede Iteration angezeigt, sondern nur für jede zehnte Iteration."
            )
            s = s_adjusted
        elif s > 1000:
            st.warning(
                "Warnung: Die Anzahl der Iterationen ist größer als 1000. "
                "Die Kosten werden dann nicht für jede Iteration angezeigt, sondern nur für jede zehnte Iteration."
            )

    # start_s
    with col6:
        start_s = st.number_input(
            "Startiteration",
            value=1,
            min_value=1,
            max_value=int(s)-1,
            help= ("Die Iteration, ab der die Kosten im Plot angezeigt werden.\n"
            "Wenn Sie die Iteration ändern und mit **ENTER** bestätigen, wird der Plot automatisch angepasst. "
                "Die Modelle müssen nicht erneut hinzugefügt werden."),
        )
        if s > 1000:
            # Bei s > 1000 nur Iteration 1 oder Vielfache von 10 sinnvoll
            if start_s != 1 and start_s % 10 != 0:
                start_adjusted = start_s - (start_s % 10)
                if start_adjusted < 10:
                    start_adjusted = 10
                st.warning(
                    f"Warnung: Bei s > 1000 wird die Startiteration auf das nächste durch 10 teilbare Niveau "
                    f"angepasst: {start_adjusted}."
                )
                start_s = start_adjusted

    # Modelle hinzufügen
    if st.button("Hinzufügen", key="add_btn"):
        sel_h, sel_M, sel_L, sel_d = (
            st.session_state["add_h"],
            st.session_state["add_M"],
            st.session_state["add_L"],
            st.session_state["add_d"],
        )
        added = 0
        for h_, M_, L_, d_ in itertools.product(sel_h, sel_M, sel_L, sel_d):
            new_spec = (h_, M_, L_, epst, d_, epo)
            label = f"h = {h_}, M = {M_}, L = {L_}, d = {d_}"
            if not any(item["spec"] == new_spec for item in st.session_state.extra_lines):
                st.session_state.extra_lines.append({"label": label, "spec": new_spec})
                added += 1

        st.session_state["_added_count"] = added
        st.session_state["_show_added"] = True
        st.rerun()

# --- Modelle entfernen ---
with st.expander("🧹 Modelle entfernen", expanded=False):
    if st.session_state.extra_lines:
        if st.button("Alle Modelle entfernen", key="clear_all", help="Alle zusätzlichen Modelle entfernen"):
            st.session_state.extra_lines.clear()
            st.rerun()
        st.divider()
        for i, item in enumerate(st.session_state.extra_lines):
            c1, c2, c3 = st.columns([1, 6, 1])
            with c1:
                st.write(f"{i+1}.")
            with c2:
                st.write(item["label"])
            with c3:
                if st.button("🗑️", key=f"del_{i}", help="Dieses Modell entfernen"):
                    st.session_state.extra_lines.pop(i)
                    st.rerun()
    else:
        st.caption("Keine zusätzlichen Linien vorhanden.")

# --- Index des Default-Modells ---
idx = [index for index, tpl in enumerate(param_list) if tpl == selected_tuple][0]

# all_losses_reduced: Spalten 1..9 löschen (angenommen: 0 -> Iteration 1, ab 10er-Schritte)
#all_losses_reduced = np.delete(all_losses_array, np.s_[1:9], axis=1)
cols_part1 = np.arange(9, 1000, 10)
cols_part2 = np.arange(1000, 10900)
cols = np.concatenate([cols_part1, cols_part2])
all_losses_reduced = all_losses_array[:, cols]

# --- Achse & y-Werte vorbereiten ---
if s > 1000:
    # Fall: nur jede 10. Iteration gespeichert (plus evtl. Iteration 1)
    if start_s == 1:
        xplot = np.arange(10, s + 10, 10, dtype=np.float32)
        xplot[0] = 1  # 0 -> 1
        start_col = 0
    else:
        # start_s ist Vielfaches von 10
        xplot = np.arange(start_s, s + 10, 10, dtype=np.float32)
        start_col = start_s // 10 - 1 # 10 -> 1, 20 -> 2, ...

    end_col = s // 10
    y_hat = all_losses_reduced[idx, start_col : end_col + 1]
else:
    # Vollständige Iterationen 1..s
    xplot = np.arange(start_s, s + 1, dtype=np.float32)
    y_hat = all_losses_array[idx, (start_s - 1) : s]

# --- Plot ---
if st.session_state.extra_lines:
    fig, ax = plt.subplots()
    ax.set_title("Die empirischen Kosten im Verlauf des Trainings", fontsize=12, pad=8)

    for item in st.session_state.extra_lines:
        spec = item["spec"]
        label = item["label"]
        idx2 = [i for i, tpl in enumerate(param_list) if tpl == spec][0]

        if s > 1000:
            if start_s == 1:
                start_col2 = 0
            else:
                start_col2 = start_s // 10 -1
            end_col2 = s // 10
            y_hat2 = all_losses_reduced[idx2, start_col2 : end_col2 + 1]
        else:
            y_hat2 = all_losses_array[idx2, start_s - 1 : s]

        ax.plot(xplot, y_hat2[: len(xplot)], label=label)

    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)

    if n > 10:
        ncol = min(5, math.ceil(n / 2))
        w, h = fig.get_size_inches()
        extra_h_in = 0.9
        fig.set_size_inches(w, h + extra_h_in, forward=True)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=False,
            fontsize=9,
            handlelength=2.0,
            columnspacing=1.2,
            handletextpad=0.6,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.22 if ncol <= 4 else 0.28)
    else:
        ax.legend(loc="best")
        plt.tight_layout()

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Empirische Kosten")

    st.pyplot(fig, width="content")

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    st.info(
        "- **h** – Schrittweite des Gradientenverfahrens\n"
        "- **M** – Breite: Anzahl der Neuronen pro Schicht\n"
        "- **L** – Tiefe: Anzahl der Schichten\n"
        "- **d** – Dimension der Trainingsdaten\n"
        f"- Es werden Iterationen im Bereich **[{int(start_s)}, {int(s)}]** angezeigt."
    )
else:
    st.warning(
        f"Fügen Sie über **{expand_label}** eine oder mehrere Netzwerkarchitekturen (Breite, Tiefe) "
        "mit zugehöriger Dimension der Trainingsdaten und Schrittweite des Gradientenverfahrens hinzu. "
        "Die Kosten (loss) werden dann im Verlauf der Trainingsiterationen angezeigt."
    )
