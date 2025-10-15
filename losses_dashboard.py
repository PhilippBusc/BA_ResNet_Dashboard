import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import itertools
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="ResNet Approximation", page_icon="📈")
st.subheader("📈 ResNet-Approximation von sinus(x)")
main_path = "Data/"
all_losses_array = joblib.load(main_path + "all_losses_array.joblib")
# Parameter options
h_values = [0.01]
stop_values = [1e-10]
M_values = [1 , 2, 4, 6, 8]
L_values = [1, 2, 4, 6, 8]
d_values = [10, 20, 2, 4, 8]
max_num_epochs = [1000000]

# All parameter combinations
param_list = list(itertools.product(h_values, M_values, L_values, stop_values, d_values, max_num_epochs))

d_values_input= [2, 4, 8, 10, 20]
# Default input   
h = h_values[0]
M = M_values[0]
L = L_values[0]
d = d_values_input[0]
s = 1000


epst  = stop_values[0]
epo   = 1000000
selected_tuple = (h, M, L, epst, d, epo) 



h_idx = h_values.index(h)
M_idx = M_values.index(M)
L_idx = L_values.index(L)
d_idx = d_values_input.index(d)

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


if "add_state" not in st.session_state:
    st.session_state.add_state = True
    label = f"h = {h}, M = {M}, L = {L}, d = {d}"
    st.session_state.extra_lines.append({"label": label, "spec": selected_tuple})

# Add expander
expand_label = "➕ Modelle hinzufügen"
with st.expander(expand_label, expanded=False):
    if st.session_state.pop("_show_added", False):
        added = st.session_state.pop("_added_count", 0)
        if added:
            st.toast(f"{added} Modell(e) hinzugefügt.", icon="✅")
        else:
            st.toast("Keine neuen Modelle (evtl. Duplikate).", icon="ℹ️")

    col1, col2, col3, col4, col5 = st.columns(5)

    
    default_h = [h_values[st.session_state.h_idx[-1]]] if st.session_state.get("h_idx") else [h_values[0]]
    default_M = [M_values[st.session_state.M_idx[-1]]] if st.session_state.get("M_idx") else [M_values[0]]
    default_L = [L_values[st.session_state.L_idx[-1]]] if st.session_state.get("L_idx") else [L_values[0]]
    default_d = [d_values_input[st.session_state.d_idx[-1]]] if st.session_state.get("d_idx") else [d_values_input[0]]

    st.session_state.setdefault("add_h", default_h)
    st.session_state.setdefault("add_M", default_M)
    st.session_state.setdefault("add_L", default_L)
    st.session_state.setdefault("add_d", default_d)

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
        add_h = st.multiselect("Schrittweite (h)", options=h_values, default=st.session_state["add_h"], key="add_h", help = "Schrittweite des Gradientenverfahrens")
        if st.button("Alle auswählen", key="all_h_btn"):
            st.session_state["_all_h_trigger"] = True
            st.rerun()

    with col2:
        add_M = st.multiselect("Breite (M)", options=M_values, default=st.session_state["add_M"], key="add_M", help = "Breite des ResNet - Anzahl der Neuronen pro Schicht")
        if st.button("Alle auswählen", key="all_M_btn"):
            st.session_state["_all_M_trigger"] = True
            st.rerun()

    with col3:
        add_L = st.multiselect("Tiefe (L)", options=L_values, default=st.session_state["add_L"], key="add_L", help = "Tiefe des ResNet - Anzahl der Schichten")
        if st.button("Alle auswählen", key="all_L_btn"):
            st.session_state["_all_L_trigger"] = True
            st.rerun()

    with col4:
        add_d = st.multiselect("Dimension (d)", options=d_values_input, default=st.session_state["add_d"], key="add_d", help = "Dimension der Trainingsdaten")
        if st.button("Alle auswählen", key="all_d_btn"):
            st.session_state["_all_d_trigger"] = True
            st.rerun()

    with col5:
        s     = st.number_input("Iterationen (s)", value= 1000, min_value= 2, max_value = max_num_epochs[0],format="%.1e", help="Anzahl der Trainingsiterationen (für alle Modelle gleich). \n Wenn Sie **s** ändern und mit ENTER bestätigen, wird der Plot automatisch angepasst. Die Modelle müssen nicht erneut hinzugefügt werden")     
        if s > 1000 and s % 1000 != 0:
            s = s - s % 1000
            st.warning("Warnung: Die Anzahl der Iterationen ist größer als 1000 und nicht durch 1000 teilbar. Die Anzahl der Iterationen wird automatisch auf f{s} gesetzt. Die Kosten werden nicht für jede Iteration angezeigt, sondern nur für jede 1000ste Iteration.")
        elif s > 1000:
            st.warning("Warnung: Die Anzahl der Iterationen ist größer als 1000. Die Kosten werden dann nicht für jede Iteration angezeigt, sondern nur für jede 1000ste Iteration.")  


    if st.button("Hinzufügen", key="add_btn"):
        import itertools
        sel_h, sel_M, sel_L, sel_d = st.session_state["add_h"], st.session_state["add_M"], st.session_state["add_L"], st.session_state["add_d"]
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

# Remove expander
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


idx = [index for index, tpl in enumerate(param_list) if tpl == selected_tuple][0]
all_losses_reduced = np.delete(all_losses_array, np.s_[1:999], axis=1)
        

if s > 1000:
    xplot  =  np.arange(0, s+1000, 1000, dtype=np.float32)
    xplot[0] = 1
    y_hat = all_losses_reduced[idx, 0:int(s/1000)+1]
else:
    xplot  =  np.arange(1, s+1, dtype=np.float32)
    y_hat = all_losses_array[idx, 0:s]


if st.session_state.extra_lines:
    fig, ax = plt.subplots()
    ax.set_title("Die Kosten (loss) im Verlauf des Trainings", fontsize=12, pad=8, weight="semibold")

    for item in st.session_state.extra_lines:
        spec = item["spec"]
        label = item["label"]
        idx2 = [i for i, tpl in enumerate(param_list) if tpl == spec][0]
        if s > 1000:
            y_hat2 = all_losses_reduced[idx2, 0:int(s/1000)+1]
        else:
            y_hat2 = all_losses_array[idx2, 0:int(s)]
        ax.plot(xplot, y_hat2[:len(xplot)], label=label)


    

    lines = ax.get_lines()
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)

    if n > 10:
        ncol = min(5, math.ceil(n / 2)) 
        w, h = fig.get_size_inches()
        extra_h_in = 0.9  
        fig.set_size_inches(w, h + extra_h_in, forward=True)
        fig.legend(
        handles, labels,
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
        

    ax.set_xlabel("Iteration"); ax.set_ylabel("Kosten (loss)")#; ax.legend()

    st.pyplot(fig, width = "content")




    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


    st.info(
        "- **h** – Schrittweite des Gradientenverfahrens\n"
        "- **M** – Breite: Anzahl der Neuronen pro Schicht\n"
        "- **L** – Tiefe: Anzahl der Schichten\n"
        "- **d** – Dimension der Trainingsdaten"
    )
else:
    st.warning(f"Fügen Sie über **{expand_label}** eine oder mehrere Netzwerkarchitekturen (Breite, Tiefe) mit zugehöriger Dimension der Trainingsdaten und Schrittweite des Gradientenverfahrens hinzu. Die Kosten (loss) werden dann im Verlauf der Trainingsiterationen angezeigt.")




