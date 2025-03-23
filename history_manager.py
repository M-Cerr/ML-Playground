import pandas as pd
import copy
from collections import deque
from datetime import datetime
import streamlit as st

MAX_HISTORY = 5

class HistoryEntry:
    """Represents a single modification: a snapshot of the DataFrame and metadata."""
    def __init__(self, df_snapshot, description, timestamp=None):
        # Use deep copy or clone to store
        # but here we assume df_snapshot is already a copy at creation time
        self.df_snapshot = df_snapshot
        self.description = description
        if timestamp is None:
            self.timestamp = datetime.now()
        else:
            self.timestamp = timestamp
        self.valid = True  # used if a re-apply becomes invalid

class DatasetHistory:
    """
    Manages a list of up to MAX_HISTORY snapshots.
    We also keep a current_index pointer to track the user's 'timeline' position.
    """
    def __init__(self, max_size=MAX_HISTORY):
        self.entries = []
        self.current_index = -1
        self.max_size = max_size

    def record_change(self, df, description):
        """
        Record a new snapshot of df with a short description.
        The new snapshot becomes the new 'latest' state, and we discard future states if any.
        """
        # If we are not at the last index, it means we have undone some states. 
        # We remove states after current_index to allow a new branch of changes.
        if self.current_index < len(self.entries) - 1:
            self.entries = self.entries[:self.current_index + 1]

        # If adding a new snapshot beyond the max history, remove from the front
        if len(self.entries) == self.max_size:
            # drop the earliest snapshot
            self.entries.pop(0)
            self.current_index -= 1

        # Create a new entry
        new_entry = HistoryEntry(
            df_snapshot=copy.deepcopy(df),  # ensure we store a safe copy
            description=description
        )
        self.entries.append(new_entry)
        self.current_index = len(self.entries) - 1

    def get_current_df(self):
        """Return the df for the current index or None if no history yet."""
        if self.current_index < 0 or self.current_index >= len(self.entries):
            return None
        return self.entries[self.current_index].df_snapshot

    def get_history_list(self):
        """Return all entries for UI display. Index 0 is earliest, last is newest."""
        return self.entries

    def revert_to_index(self, idx):
        """
        Move current index pointer to idx, meaning we revert the dataset state to that snapshot.
        """
        if 0 <= idx < len(self.entries):
            self.current_index = idx
        # else out of range => ignore

    def mark_invalid(self, idx):
        """If re-applying a future step is not feasible, mark it invalid."""
        if 0 <= idx < len(self.entries):
            self.entries[idx].valid = False

    def can_reapply(self, idx):
        """Check if an entry is valid for re-apply or if it's out of range."""
        if 0 <= idx < len(self.entries):
            return self.entries[idx].valid
        return False

def normalize_dataset_name(name):
    return name.replace(" ", "").lower()

def record_new_change(dataset_name, new_df, description):
    """
    Calls the current History using dataset_name, and then records changes that lead to new_df along with
    description. Used many times, so recorded here for convenience. 
    """
    normalized_name = normalize_dataset_name(dataset_name)
    if normalized_name not in st.session_state.get("histories", {}):
        st.session_state.setdefault("histories", {})[normalized_name] = DatasetHistory(max_size=5)
    history_obj = st.session_state["histories"][normalized_name]
    history_obj.record_change(new_df, description)
