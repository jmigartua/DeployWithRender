import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib import rc
import dash_bootstrap_components as dbc

# Ensure Matplotlib uses a non-interactive backend for Dash
matplotlib.use('Agg')

# Use LaTeX for text rendering
rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dcc.Store(id='store-column-defs', data=[]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=dbc.Button("Upload CSV File", color='primary', className='mb-2'),
                multiple=False
            ),
            dbc.Button("Open Empty Grid", id='open-empty-grid', color='secondary', className='mb-2'),
            dbc.Button("Add Column", id='add-column-button', color='info', className='mb-2'),
            dbc.Button("Add Row", id='add-row-button', color='info', className='mb-2'),
        ], width=2),
        dbc.Col([
            dag.AgGrid(
                id='data-grid',
                columnDefs=[],
                rowData=[],
                defaultColDef={'flex': 1, 'sortable': True, 'filter': True, 'resizable': True, 'editable': True},
                dashGridOptions={'rowSelection': 'multiple'},
                style={'height': '400px', 'width': '100%'}
            )
        ], width=10),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select X-axis:'),
            dcc.Dropdown(id='x-axis-dropdown', options=[], clearable=False)
        ], width=6),
        dbc.Col([
            html.Label('Select Y-axis:'),
            dcc.Dropdown(id='y-axis-dropdown', options=[], clearable=False)
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Plot All Points (Left)', id='plot-all-button-left', color='primary', className='mb-2'),
            dbc.Button('Plot Selected Points (Left)', id='plot-button-left', color='primary', className='mb-2'),
            dbc.Button('Perform Linear Regression (Left)', id='regression-button-left', color='danger', className='mb-2'),
            dbc.Button('Generate Matplotlib Plot (Left)', id='matplotlib-plot-button-left', color='warning', className='mb-2'),
            dcc.Graph(id='plot-left', config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}),
            html.Img(id='matplotlib-plot-left', style={'display': 'none', 'width': '100%', 'margin': '0 auto', 'text-align': 'center'}),
            html.Div(id='regression-results-left', style={'textAlign': 'center'}),
            dbc.Button('Export Plot (Left)', id='export-button-left', color='success', className='mb-2'),
            dcc.Download(id="download-plot-left")
        ], width=6),
        dbc.Col([
            dbc.Button('Plot Selected Points (Right)', id='plot-button-right', color='primary', className='mb-2'),
            dbc.Button('Perform Linear Regression (Right)', id='regression-button-right', color='danger', className='mb-2'),
            dbc.Button('Generate Matplotlib Plot (Right)', id='matplotlib-plot-button-right', color='warning', className='mb-2'),
            dbc.Button('Reset Right Plot', id='reset-button-right', color='secondary', className='mb-2'),
            dcc.Graph(id='plot-right', config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}),
            html.Img(id='matplotlib-plot-right', style={'display': 'none', 'width': '100%', 'margin': '0 auto', 'text-align': 'center'}),
            html.Div(id='regression-results-right', style={'textAlign': 'center'}),
            dbc.Button('Export Plot (Right)', id='export-button-right', color='success', className='mb-2'),
            dcc.Download(id="download-plot-right")
        ], width=6),
    ])
], fluid=True)

# Store the initial state of the left plot
initial_left_state = {'data': [], 'layout': {}}

@app.callback(
    Output('data-grid', 'columnDefs'),
    Output('data-grid', 'rowData'),
    Output('x-axis-dropdown', 'options'),
    Output('y-axis-dropdown', 'options'),
    Output('store-column-defs', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('open-empty-grid', 'n_clicks'),
    Input('add-column-button', 'n_clicks'),
    Input('add-row-button', 'n_clicks'),
    State('data-grid', 'columnDefs'),
    State('data-grid', 'rowData')
)
def update_table(contents, filename, open_empty_clicks, add_column_clicks, add_row_clicks, column_defs, row_data):
    ctx_trigger = ctx.triggered_id
    if ctx_trigger == 'upload-data' and contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        columns = [{"headerName": col, "field": col, "editable": True} for col in df.columns]
        data = df.to_dict('records')
        
        options = [{'label': col, 'value': col} for col in df.columns]
        return columns, data, options, options, columns
    
    if ctx_trigger == 'open-empty-grid':
        columns = [{"headerName": f"Column {i+1}", "field": f"Column {i+1}", "editable": True} for i in range(6)]
        data = [{"Column 1": "", "Column 2": "", "Column 3": "", "Column 4": "", "Column 5": "", "Column 6": ""} for _ in range(10)]
        options = [{'label': f"Column {i+1}", 'value': f"Column {i+1}"} for i in range(6)]
        return columns, data, options, options, columns

    if ctx_trigger == 'add-column-button':
        new_col_index = len(column_defs) + 1
        new_col = {"headerName": f"Column {new_col_index}", "field": f"Column {new_col_index}", "editable": True}
        column_defs.append(new_col)
        for row in row_data:
            row[f"Column {new_col_index}"] = ""
        options = [{'label': col['headerName'], 'value': col['field']} for col in column_defs]
        return column_defs, row_data, options, options, column_defs

    if ctx_trigger == 'add-row-button':
        new_row = {col['field']: "" for col in column_defs}
        row_data.append(new_row)
        return column_defs, row_data, dash.no_update, dash.no_update, column_defs

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('plot-left', 'figure'),
    Output('plot-right', 'figure'),
    Input('plot-button-left', 'n_clicks'),
    Input('plot-all-button-left', 'n_clicks'),
    State('data-grid', 'selectedRows'),
    State('data-grid', 'rowData'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value')
)
def update_plots(plot_selected_clicks, plot_all_clicks, selected_rows, all_rows, x_col, y_col):
    ctx_trigger = ctx.triggered_id

    if not x_col or not y_col:
        return go.Figure(), go.Figure()

    if ctx_trigger == 'plot-button-left' and selected_rows:
        df = pd.DataFrame(selected_rows)
    elif ctx_trigger == 'plot-all-button-left':
        df = pd.DataFrame(all_rows)
    else:
        return go.Figure(), go.Figure()

    try:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    except ValueError:
        return go.Figure(), go.Figure()

    df.dropna(subset=[x_col, y_col], inplace=True)

    fig_left = px.scatter(df, x=x_col, y=y_col)
    fig_right = px.scatter(df, x=x_col, y=y_col)

    # Store the initial state of the left plot
    global initial_left_state
    initial_left_state = fig_left.to_dict()

    return fig_left, fig_right

@app.callback(
    Output('plot-left', 'figure', allow_duplicate=True),
    Input('regression-button-left', 'n_clicks'),
    State('plot-left', 'selectedData'),
    State('plot-left', 'figure'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    prevent_initial_call=True
)
def perform_regression_left(n_clicks_left, selected_data_left, plot_left, x_col, y_col):
    if selected_data_left and 'points' in selected_data_left and len(selected_data_left['points']) >= 2:
        points = selected_data_left['points']
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
    else:
        x = np.array(plot_left['data'][0]['x'], dtype=float)
        y = np.array(plot_left['data'][0]['y'], dtype=float)

    coeffs = np.polyfit(x, y, 1)
    line = coeffs[0] * x + coeffs[1]

    fig = go.Figure(data=[
        go.Scatter(x=x, y=y, mode='markers', name='Data points'),
        go.Scatter(x=x, y=line, mode='lines', name='Fit')
    ])
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

@app.callback(
    Output('plot-right', 'figure', allow_duplicate=True),
    Input('regression-button-right', 'n_clicks'),
    State('plot-right', 'selectedData'),
    State('plot-right', 'figure'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    prevent_initial_call=True
)
def perform_regression_right(n_clicks_right, selected_data_right, plot_right, x_col, y_col):
    if selected_data_right and 'points' in selected_data_right and len(selected_data_right['points']) >= 2:
        points = selected_data_right['points']
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
    else:
        x = np.array(plot_right['data'][0]['x'], dtype=float)
        y = np.array(plot_right['data'][0]['y'], dtype=float)

    coeffs = np.polyfit(x, y, 1)
    line = coeffs[0] * x + coeffs[1]

    fig = go.Figure(data=[
        go.Scatter(x=x, y=y, mode='markers', name='Data points'),
        go.Scatter(x=x, y=line, mode='lines', name='Fit')
    ])
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

@app.callback(
    Output('plot-right', 'figure', allow_duplicate=True),
    Input('reset-button-right', 'n_clicks'),
    prevent_initial_call=True
)
def reset_right_plot(n_clicks):
    global initial_left_state
    fig_right = go.Figure(data=initial_left_state['data'])
    fig_right.update_layout(
        xaxis_title=initial_left_state['layout']['xaxis']['title']['text'],
        yaxis_title=initial_left_state['layout']['yaxis']['title']['text']
    )
    return fig_right

def generate_matplotlib_plot(plot, selected_data, x_col, y_col):
    if not plot or not x_col or not y_col:
        return dash.no_update, {'display': 'none'}, ""

    if selected_data and 'points' in selected_data and len(selected_data['points']) >= 2:
        points = selected_data['points']
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
    else:
        x = np.array(plot['data'][0]['x'], dtype=float)
        y = np.array(plot['data'][0]['y'], dtype=float)

    # Perform linear regression using np.polyfit
    p, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    line = slope * x + intercept

    fig = Figure()
    ax = fig.subplots()
    ax.scatter(x, y, label='Data points')
    ax.plot(x, line, color='red', label='Linear fit')
    ax.set_xlabel(x_col, fontsize=15)
    ax.set_ylabel(y_col, fontsize=15)
    ax.legend()

    # Apply the styling changes
    ax.tick_params(axis='y', labelsize=15, pad=10, length=12)
    ax.tick_params(axis='x', labelsize=15, pad=10, length=12)

    # Construct the LaTeX strings for the equation, slope, and intercept
    equation = r'$y = m \cdot x + n$'
    slope_text = r'$m \pm \Delta m = {:.4f} \pm {:.4f}$'.format(slope, slope_err)
    intercept_text = r'$n \pm \Delta n = {:.4f} \pm {:.4f}$'.format(intercept, intercept_err)
    
    # Add the texts below the plot
    plt.subplots_adjust(bottom=0.45)  # Further increase bottom margin to avoid overlap
    plt.figtext(0.5, 0.25, equation, ha='center', fontsize=15)
    plt.figtext(0.5, 0.17, slope_text, ha='center', fontsize=15)
    plt.figtext(0.5, 0.10, intercept_text, ha='center', fontsize=15)

    # Convert plot to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    buf.close()
    
    # Display LaTeX text as image
    latex_fig, latex_ax = plt.subplots(figsize=(6, 1))
    latex_ax.axis('off')
    latex_text = '\n'.join([equation, slope_text, intercept_text])
    latex_ax.text(0.5, 0.5, latex_text, ha='center', va='center', fontsize=15, usetex=True)
    
    latex_buf = io.BytesIO()
    latex_fig.savefig(latex_buf, format='png', bbox_inches='tight', pad_inches=0.1)
    latex_buf.seek(0)
    latex_data = base64.b64encode(latex_buf.read()).decode()
    latex_buf.close()
    
    latex_img = html.Img(src=f"data:image/png;base64,{latex_data}", style={'display': 'block', 'margin': '0 auto'})
    
    return f"data:image/png;base64,{plot_data}", {'display': 'block', 'margin': '0 auto', 'text-align': 'center'}, latex_img

@app.callback(
    Output('matplotlib-plot-left', 'src'),
    Output('matplotlib-plot-left', 'style'),
    Output('regression-results-left', 'children'),
    Input('matplotlib-plot-button-left', 'n_clicks'),
    State('plot-left', 'figure'),
    State('plot-left', 'selectedData'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value')
)
def generate_matplotlib_plot_left(n_clicks, plot_left, selected_data, x_col, y_col):
    return generate_matplotlib_plot(plot_left, selected_data, x_col, y_col)

@app.callback(
    Output('matplotlib-plot-right', 'src'),
    Output('matplotlib-plot-right', 'style'),
    Output('regression-results-right', 'children'),
    Input('matplotlib-plot-button-right', 'n_clicks'),
    State('plot-right', 'figure'),
    State('plot-right', 'selectedData'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value')
)
def generate_matplotlib_plot_right(n_clicks, plot_right, selected_data, x_col, y_col):
    return generate_matplotlib_plot(plot_right, selected_data, x_col, y_col)

def export_plot(plot, x_col, y_col):
    if not plot or not x_col or not y_col:
        return

    x = np.array(plot['data'][0]['x'], dtype=float)
    y = np.array(plot['data'][0]['y'], dtype=float)

    # Perform linear regression using np.polyfit
    p, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    line = slope * x + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, label='Data points')
    ax.plot(x, line, color='red', label='Linear fit')
    ax.set_xlabel(x_col, fontsize=15)
    ax.set_ylabel(y_col, fontsize=15)
    ax.legend()

    # Apply the styling changes
    ax.tick_params(axis='y', labelsize=15, pad=10, length=12)
    ax.tick_params(axis='x', labelsize=15, pad=10, length=12)

    # Construct the LaTeX strings for the equation, slope, and intercept
    equation = r'$y = m \cdot x + n$'
    slope_text = r'$m \pm \Delta m = {:.4f} \pm {:.4f}$'.format(slope, slope_err)
    intercept_text = r'$n \pm \Delta n = {:.4f} \pm {:.4f}$'.format(intercept, intercept_err)
    
    # Add the texts below the plot
    plt.subplots_adjust(bottom=0.45)  # Further increase bottom margin to avoid overlap
    plt.figtext(0.5, 0.25, equation, ha='center', fontsize=15)
    plt.figtext(0.5, 0.17, slope_text, ha='center', fontsize=15)
    plt.figtext(0.5, 0.10, intercept_text, ha='center', fontsize=15)

    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', dpi=200)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode()
    buf.close()
    
    return dict(content=plot_data, filename="plot.pdf", type="application/pdf", base64=True)

@app.callback(
    Output("download-plot-left", "data"),
    Input('export-button-left', 'n_clicks'),
    State('plot-left', 'figure'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    prevent_initial_call=True
)
def export_plot_left(n_clicks, plot_left, x_col, y_col):
    return export_plot(plot_left, x_col, y_col)

@app.callback(
    Output("download-plot-right", "data"),
    Input('export-button-right', 'n_clicks'),
    State('plot-right', 'figure'),
    State('x-axis-dropdown', 'value'),
    State('y-axis-dropdown', 'value'),
    prevent_initial_call=True
)
def export_plot_right(n_clicks, plot_right, x_col, y_col):
    return export_plot(plot_right, x_col, y_col)

if __name__ == '__main__':
    app.run_server(debug=True)
