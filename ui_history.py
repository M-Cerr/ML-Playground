import streamlit as st
from datetime import datetime
from history_manager import DatasetHistory

def normalize_dataset_name(name):
    return name.replace(" ", "").lower()

# Define the history modal as a function decorated with st.dialog.
@st.dialog("History", width="large")
def history_modal_fn():
    # Retrieve dataset name from session state.
    dataset_name = st.session_state.get("history_modal_dataset")
    if not dataset_name:
        st.write("No dataset selected for history.")
        return

    st.write("## Modification History")
    st.write("Below are the last modifications. Undo or re-apply as needed.")
    _render_history_list(dataset_name)
    # When the Close button is clicked, remove the modal trigger.
    if st.button("Close History"):
        st.session_state.pop("history_modal_dataset", None)
        st.rerun()

def display_history_ui(dataset_name):
    """
    Displays the floating 'History' button. When clicked, it directly opens the modal dialog.
    """

    #st.write("DEBUG: display_history_ui called for", dataset_name)

    normalized = normalize_dataset_name(dataset_name)
    # Store the normalized dataset name in session state for modal purposes.
    st.session_state["current_history_dataset"] = normalized

    # Retrieve or create the persistent placeholder for the floating button.
    if "history_button_placeholder" not in st.session_state:
        st.session_state["history_button_placeholder"] = st.empty()
    placeholder = st.session_state["history_button_placeholder"]

    placeholder.markdown(f"""
    <style>
    /* We style the button associated with key="historyfloating" */
    .st-key-historyfloating_{normalized} button {{
        position: fixed;
        top: 70px;
        right: 30px;
        background-color: #2196F3;
        color: white;
        padding: 10px 16px;
        border: none;
        border-radius: 4px;
        z-index: 9998;
        cursor: pointer;
    }}
    .st-key-historyfloating_{normalized} button:hover {{
        background-color: #0b7dda;
    }}
    </style>
    """, unsafe_allow_html=True)

    # This hidden button triggers show_history_modal
    if st.button("Open History", key=f"historyfloating_{normalized}"):
        st.session_state["history_modal_dataset"] = normalized
        history_modal_fn()


def _render_history_list(dataset_name):
    """Renders the list of modifications from st.session_state["histories"][dataset_name]."""
    if "histories" not in st.session_state or dataset_name not in st.session_state["histories"]:
        st.write("No history found for this dataset.")
        return

    history_obj = st.session_state["histories"][dataset_name]
    entries = history_obj.get_history_list()
    if not entries:
        st.write("No modifications yet.")
        return

    current_idx = history_obj.current_index

    # List modifications from newest to oldest.
    for i in reversed(range(len(entries))):
        entry = entries[i]
        col1, col2 = st.columns([4, 2])
        with col1:
            ts_str = entry.timestamp.strftime('%H:%M:%S')
            st.write(f"**[{ts_str}]** {entry.description}")
        with col2:
            if i == current_idx:
                st.write(":white_check_mark: (Current state)")
            elif i < current_idx:
                if st.button("Undo", key=f"undo_{i}"):
                    history_obj.revert_to_index(i)
                    df_reverted = history_obj.get_current_df()
                    st.session_state['datasets'][dataset_name] = df_reverted
                    st.session_state["last_modification"] = f"Reverted to modification #{i}."
                    st.rerun()
            else:
                if entry.valid:
                    if st.button("Re-Apply", key=f"redo_{i}"):
                        history_obj.revert_to_index(i)
                        df_restored = history_obj.get_current_df()
                        st.session_state['datasets'][dataset_name] = df_restored
                        st.session_state["last_modification"] = f"Re-applied modification #{i}."
                        st.rerun()
                else:
                    st.write(":x: Not re-applicable")