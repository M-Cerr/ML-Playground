import streamlit as st

def display_section_header_with_help(title: str, expander_title: str, help_text: str, key: str):
    """
    Displays a section header with a small help expander next to it.
    
    Parameters:
      - title: The title for the section.
      - help_text: The explanation text to show when expanded.
      - key: A unique key for the expander.
    """
    # Use two columns: first for the subheader, second for the help expander.
    # Adjust the ratio as needed (here 8:2 gives a narrow column for the expander).
    col1, col2, col3 = st.columns([3, 4, 3])
    with col1:
        st.subheader(title)
    with col2:
        # We force the expander to align its content to the left by adding some padding.
        with st.expander(f"{expander_title}", expanded=False):
            st.markdown(f"<div style='padding-left: 5px;'>{help_text}</div>", unsafe_allow_html=True)