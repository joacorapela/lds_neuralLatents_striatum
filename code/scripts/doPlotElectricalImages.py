import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# --- Data Loading (same as before) ---
def load_ks_spatial(data_path):
    path = Path(data_path)
    templates = np.load(path / 'templates.npy')
    whitening_inv = np.load(path / 'whitening_mat_inv.npy')
    ch_pos = np.load(path / 'channel_positions.npy')
    return np.matmul(templates, whitening_inv), ch_pos

# Replace with your path
DATA_PATH = "/ceph/sjones/projects/sequence_squad/data/kilosort_output/EJT178/Recording6/2022-03-29_13-55-06/kilosort3"
all_templates, ch_pos = load_ks_spatial(DATA_PATH)
n_temps, n_samples, _ = all_templates.shape

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Template Spatial Explorer", style={'textAlign': 'center'}),
    
    # CONTROL PANEL
    html.Div([
        # Row 1: Template and Time
        html.Div([
            html.Label("Template ID:"),
            dcc.Dropdown(id='template-sel', options=[{'label': i, 'value': i} for i in range(n_temps)], value=0),
            html.Label("Time Sample:"),
            dcc.Slider(id='time-slider', min=0, max=n_samples-1, step=1, value=n_samples//2),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Row 2: Axis Limits
        html.Div([
            html.Label("Coordinate Limits (μm):"),
            html.Div([
                dcc.Input(id='x-min', type='number', placeholder='X Min', style={'width': '22%'}),
                dcc.Input(id='x-max', type='number', placeholder='X Max', style={'width': '22%'}),
                dcc.Input(id='y-min', type='number', placeholder='Y Min', style={'width': '22%'}),
                dcc.Input(id='y-max', type='number', placeholder='Y Max', style={'width': '22%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),

    dcc.Graph(id='spatial-map', style={'height': '75vh'})
])

@app.callback(
    Output('spatial-map', 'figure'),
    [Input('template-sel', 'value'),
     Input('time-slider', 'value'),
     Input('x-min', 'value'), Input('x-max', 'value'),
     Input('y-min', 'value'), Input('y-max', 'value')]
)
def update_plot(tid, t_idx, xmin, xmax, ymin, ymax):
    voltages = all_templates[tid, t_idx, :]
    v_max = np.max(np.abs(all_templates[tid]))

    fig = go.Figure(data=go.Scatter(
        x=ch_pos[:, 0], y=ch_pos[:, 1],
        mode='markers',
        marker=dict(
            size=10, color=voltages, colorscale='RdBu_r',
            cmin=-v_max, cmax=v_max, colorbar=dict(title="μV"),
            symbol='square', line=dict(width=0.5, color='gray')
        ),
        text=[f"Ch {i}: {v:.2f}μV" for i, v in enumerate(voltages)],
        hoverinfo='text'
    ))

    # Apply Manual Range if provided, else use Auto
    fig.update_xaxes(range=[xmin, xmax] if xmin is not None and xmax is not None else None)
    fig.update_yaxes(range=[ymin, ymax] if ymin is not None and ymax is not None else None)

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Width (μm)", scaleanchor="y"),
        yaxis=dict(title="Depth (μm)"),
        margin=dict(l=40, r=40, t=40, b=40),
        uirevision=tid # Keeps zoom even if you change time, but resets if you change template
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
