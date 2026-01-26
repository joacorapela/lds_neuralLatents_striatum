import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# --- Load and Unwhiten Data ---
def load_ks_spatial(data_path):
    path = Path(data_path)
    templates = np.load(path / 'templates.npy')             # (n_templates, n_samples, n_chans)
    whitening_inv = np.load(path / 'whitening_mat_inv.npy') # (n_chans, n_chans)
    channel_pos = np.load(path / 'channel_positions.npy')    # (n_chans, 2)
    
    # Pre-unwhiten for the app
    unwhitened_all = np.matmul(templates, whitening_inv)
    return unwhitened_all, channel_pos

DATA_PATH = "/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3"
all_templates, ch_pos = load_ks_spatial(DATA_PATH)
n_temps, n_samples, n_chans = all_templates.shape

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Template Spatial Map", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Template ID:"),
            dcc.Dropdown(
                id='template-sel',
                options=[{'label': f'Template {i}', 'value': i} for i in range(n_temps)],
                value=0
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Time Sample (0-81):"),
            dcc.Slider(
                id='time-slider',
                min=0, max=n_samples-1, step=1, value=n_samples//2,
                marks={i: str(i) for i in range(0, n_samples, 10)},
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'padding': '20px'}),

    dcc.Graph(id='spatial-map', style={'height': '80vh'})
])

@app.callback(
    Output('spatial-map', 'figure'),
    [Input('template-sel', 'value'),
     Input('time-slider', 'value')]
)
def update_spatial_plot(template_id, time_idx):
    # Extract voltages for ALL electrodes at the specific time point
    voltages = all_templates[template_id, time_idx, :]
    
    # Determine color limits based on the peak of the whole template for consistency
    v_max = np.max(np.abs(all_templates[template_id]))
    
    fig = go.Figure(data=go.Scatter(
        x=ch_pos[:, 0],
        y=ch_pos[:, 1],
        mode='markers',
        marker=dict(
            size=12,
            color=voltages,
            colorscale='RdBu_r', # Red for positive, Blue for negative
            cmin=-v_max,
            cmax=v_max,
            colorbar=dict(title="Voltage"),
            symbol='square' # Neuropixel sites are square
        ),
        text=[f"Ch {i}<br>Volts: {v:.4f}" for i, v in enumerate(voltages)],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"Spatial Map: Template {template_id} at Sample {time_idx}",
        xaxis=dict(title="Probe Width (μm)", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Probe Depth (μm)"),
        template="plotly_dark", # Dark mode makes colors pop
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)
