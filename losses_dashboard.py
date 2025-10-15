import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

st.set_page_config(page_title="ResNet Approximation", page_icon="📈")
st.subheader("📈 ResNet-Approximation von sinus(x)")
#main_path = "C:/Users/admin/Desktop/streamlit_test/"
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
# Input

# with st.expander("🧠 Wähle eine Netzwerkarchitektur aus", expanded=True):
#     col1, col2, col3, col4, col5  = st.columns(5)
#     with col1:
#         h     = st.selectbox("Schrittweite (h)", options=h_values, index= 0, help="Schrittweite des Gradientenverfahrens")
#     with col2:
#         M     = st.selectbox("Breite (M)", options=M_values, index=0, help="Breite des ResNet - Anzahl der Neuronen pro Schicht")
#     with col3:
#         L     = st.selectbox("Tiefe (L)", options=L_values, index=0, help = "Tiefe des ResNet - Anzahl der Schichten")
#     with col4:
#         d     = st.selectbox("Dimension (d)", options=d_values_input, index=0, help=  "Dimension der Trainingsdaten")
#     with col5: 
#         s     = st.number_input("Iterationen (s)", value= 1000, min_value= 2, max_value = max_num_epochs[0],format="%.1e", help="Anzahl der Trainingsiterationen")
    
h = h_values[0]
M = M_values[0]
L = L_values[0]
d = d_values_input[0]
s = 1000


epst  = stop_values[0]

# d_values = [2, 4, 8, 10, 20]
# h     = st.select_slider("h", options=h_values)
# epst  = stop_values[0]
# M     = st.select_slider("M", options=M_values)
# L     = st.select_slider("L", options=L_values)
# d     = st.select_slider("d", options=d_values)

epo   = 1000000

# Selection of g(x)
selected_tuple = (h, M, L, epst, d, epo) 



h_idx = int(np.where(np.array(h_values) == h)[0])
M_idx = int(np.where(np.array(M_values) == M)[0])
L_idx = int(np.where(np.array(L_values) == L)[0])
d_idx = int(np.where(np.array(d_values_input) == d)[0])

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
# --- minimal: state + small "+" panel to add extra lines ---
if "extra_lines" not in st.session_state:
    st.session_state.extra_lines = []  # each item: {"label": str, "spec": tuple}


if "add_state" not in st.session_state:
    st.session_state.add_state = True
    label = f"h = {h}, M = {M}, L = {L}, d = {d}"
    st.session_state.extra_lines.append({"label": label, "spec": selected_tuple})

     
expand_label = "➕ Modelle hinzufügen"
with st.expander(expand_label, expanded=False):
    # Show a toast once after rerun, then clear flags (prevents flashing raw text)
    if st.session_state.pop("_show_added", False):
        added = st.session_state.pop("_added_count", 0)
        if added:
            st.toast(f"{added} Modell(e) hinzugefügt.", icon="✅")
        else:
            st.toast("Keine neuen Modelle (evtl. Duplikate).", icon="ℹ️")

    col1, col2, col3, col4, col5 = st.columns(5)

    # --- sensible defaults based on last-used indices or first option ---
    default_h = [h_values[st.session_state.h_idx[-1]]] if st.session_state.get("h_idx") else [h_values[0]]
    default_M = [M_values[st.session_state.M_idx[-1]]] if st.session_state.get("M_idx") else [M_values[0]]
    default_L = [L_values[st.session_state.L_idx[-1]]] if st.session_state.get("L_idx") else [L_values[0]]
    default_d = [d_values_input[st.session_state.d_idx[-1]]] if st.session_state.get("d_idx") else [d_values_input[0]]

    # Ensure session defaults exist BEFORE widgets
    st.session_state.setdefault("add_h", default_h)
    st.session_state.setdefault("add_M", default_M)
    st.session_state.setdefault("add_L", default_L)
    st.session_state.setdefault("add_d", default_d)

    # Handle "Alle auswählen" triggers BEFORE rendering widgets (prevents SessionState errors)
    for trigger_key, target_key, all_vals in [
        ("_all_h_trigger", "add_h", h_values),
        ("_all_M_trigger", "add_M", M_values),
        ("_all_L_trigger", "add_L", L_values),
        ("_all_d_trigger", "add_d", d_values_input),
    ]:
        if st.session_state.get(trigger_key):
            st.session_state[target_key] = all_vals
            st.session_state[trigger_key] = False  # clear trigger

    # --- Widgets + buttons BELOW each multiselect ---
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


    # Add all selected combinations
    if st.button("Hinzufügen", key="add_btn"):
        import itertools
        sel_h, sel_M, sel_L, sel_d = st.session_state["add_h"], st.session_state["add_M"], st.session_state["add_L"], st.session_state["add_d"]
        added = 0
        for h_, M_, L_, d_ in itertools.product(sel_h, sel_M, sel_L, sel_d):
            new_spec = (h_, M_, L_, epst, d_, epo)  # uses epst/epo from outer scope
            label = f"h = {h_}, M = {M_}, L = {L_}, d = {d_}"
            if not any(item["spec"] == new_spec for item in st.session_state.extra_lines):
                st.session_state.extra_lines.append({"label": label, "spec": new_spec})
                added += 1

        # Defer message to after rerun (no flashing)
        st.session_state["_added_count"] = added
        st.session_state["_show_added"] = True
        st.rerun()




###############################
# --- minimal: panel to remove previously added lines ---
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
# Construct x/y as in your snippet
all_losses_reduced = np.delete(all_losses_array, np.s_[1:999], axis=1)
        

if s > 1000:
    xplot  =  np.arange(0, s+1000, 1000, dtype=np.float32)
    xplot[0] = 1
    y_hat = all_losses_reduced[idx, 0:int(s/1000)+1]
else:
    xplot  =  np.arange(1, s+1, dtype=np.float32)
    y_hat = all_losses_array[idx, 0:s]




# --- minimal: plot any extra lines the user added ---
if st.session_state.extra_lines:
    fig, ax = plt.subplots()
    #ax.plot(xplot, y_hat[:len(xplot)], label="h = {}, M = {}, L = {}, d = {}".format(h, M, L, d))
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


    import math

    # after you've plotted all lines on ax …
    lines = ax.get_lines()
    #n = len(lines)
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)

    if n > 10:
        # how many columns to keep it compact
        ncol = min(5, math.ceil(n / 2))  # e.g., 12 lines -> 6x2 or 5 columns max
        w, h = fig.get_size_inches()
        extra_h_in = 0.9  # space just for the legend; tweak if needed
        fig.set_size_inches(w, h + extra_h_in, forward=True)
        fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),      # 2% above bottom of the figure
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.6,
        )
        # add bottom space so the legend fits
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.22 if ncol <= 4 else 0.28)
    else:
        ax.legend(loc="best")
        plt.tight_layout()
        

    ax.set_xlabel("Iteration"); ax.set_ylabel("Kosten (loss)")#; ax.legend()

    st.pyplot(fig, use_container_width=False)



    # (Optional) small padding tweak
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # Simple option: Streamlit's built-in info box
    st.info(
        "- **h** – Schrittweite des Gradientenverfahrens\n"
        "- **M** – Breite: Anzahl der Neuronen pro Schicht\n"
        "- **L** – Tiefe: Anzahl der Schichten\n"
        "- **d** – Dimension der Trainingsdaten"
    )
else:
    st.warning(f"Fügen Sie über **{expand_label}** eine oder mehrere Netzwerkarchitekturen (Breite, Tiefe) mit zugehöriger Dimension der Trainingsdaten und Schrittweite des Gradientenverfahrens hinzu. Die Kosten (loss) werden dann im Verlauf der Trainingsiterationen angezeigt.")




