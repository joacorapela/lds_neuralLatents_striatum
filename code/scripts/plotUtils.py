
import numpy as np
import pandas as pd


def build_events_df(start_time_sec, end_time_sec, transition_data,
                    ports_linetypes, ports_colors,
                    start_poke_in_time_col_name="Start_Poke_in_time",
                    start_poke_out_time_col_name="Start_Poke_out_time",
                    start_port_col_name="Start_Port"):
    mask = np.logical_and(
        start_time_sec<=transition_data[start_poke_in_time_col_name],
        transition_data[start_poke_in_time_col_name]<end_time_sec
    )
    subset_transition_data = transition_data[mask]
    event_time = subset_transition_data[start_poke_in_time_col_name]
    ports_names = subset_transition_data[start_port_col_name]
    event_line_type = [ports_linetypes[port_name]
                       for port_name in ports_names]
    event_color = [ports_colors[port_name]
                   for port_name in ports_names]
    answer = pd.DataFrame(dict(event_time=event_time,
                               event_line_type=event_line_type,
                               event_color=event_color))
    return answer

