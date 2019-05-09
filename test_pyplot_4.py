import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import math
from utilities import get_color, color_string
import pandas as pd
import plotly.graph_objs as go
import plotly
import time
from pathlib import Path
from cool_data_analysis import Runs, runs_to_csvs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# runs_to_csvs(Path() / 'storage' / 'final' / 'formal1')
# runs = Runs(Path() / 'storage' / 'final' / 'formal1', done_hours=20, done_steps=100000)(game='base', han=None)
# groups = runs.group(['v'], times=False)
#
# runs_to_csvs(Path() / 'storage' / 'final' / 'formal2')
# runs = Runs(Path() / 'storage' / 'final' / 'formal2', done_hours=20, done_steps=100000)(game=('base', 'beac', '4ext'))
# groups = runs.group(['v'], times=True)

# runs_to_csvs(Path() / 'storage' / 'final' / 'formal4')
runs = Runs(Path() / 'storage' / 'final' / 'formal3', done_hours=20, done_steps=10000)
groups = runs.group(['v'], times=False)

# runs_to_csvs(Path() / 'storage' / 'graphs' / 'formal1_only_blind')
# runs = Runs(Path() / 'storage' / 'graphs' / 'formal1_only_blind', done_hours=20, done_steps=100000)
# groups = runs.group(['v'], times=False)

# runs_to_csvs(Path() / 'storage' / 'graphs' / 'formal2')
# runs = Runs(Path() / 'storage' / 'graphs' / 'formal2', done_hours=20, done_steps=100000)
# groups = runs.group(['v'], times=True)


print(groups)
i_color = 0

available_indicators = runs.runs[0].df.columns
available_group_names = [group.name for group in groups]
print(available_indicators)
print(available_group_names)


def generate_control_id(setting):
    return 'filter-{}'.format(setting)


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('width: ', style={'display': 'inline-block'}),
            html.Div(id='width-output', style={'display': 'inline-block'}),
            html.Div(dcc.Slider(id='width', min=1, max=1000, value=700, step=1), style={'float': 'right', 'flex': '1'})
        ], style={'display': 'flex', 'height': '30px'}),
        html.Div([
            html.Label('height: ', style={'display': 'inline-block'}),
            html.Div(id='height-output', style={'display': 'inline-block'}),
            html.Div(dcc.Slider(id='height', min=1, max=1000, value=800, step=1), style={'float': 'right', 'flex': '1'})
        ], style={'display': 'flex', 'height': '30px'}),
        html.Div([
            html.Label('line width: ', style={'display': 'inline-block'}),
            html.Div(dcc.Slider(id='line-width', min=0.5, marks={i / 2: i / 2 for i in range(1, 10)}, max=5, value=1.5, step=0.5), style={'float': 'right', 'flex': '1'})
        ], style={'display': 'flex', 'height': '40px'}),
        html.Div([
            html.Label('shade: ', style={'display': 'inline-block'}),
            html.Div(dcc.Slider(id='shade', min=0, marks={i / 10: i / 10 for i in range(5)}, max=0.5, value=0.15, step=0.05), style={'float': 'right', 'flex': '1'})
        ], style={'display': 'flex', 'height': '40px'}),

        html.Div([
            dcc.RadioItems(
                id='zoom',
                options=[{'label': str(i), 'value': i} for i in [0.1, 1, 10, 100]],
                value=1,
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='xzoom',
                options=[{'label': 'x1', 'value': 1}, {'label': 'x2', 'value': 2}],
                value=1,
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='diamonds',
                options=[{'label': 'diamonds', 'value': True}, {'label': 'no diamonds', 'value': False}],
                value=False,
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='cards',
                options=[{'label': 'cards', 'value': True}, {'label': 'no cards', 'value': False}],
                value=True,
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='legend',
                options=[{'label': 'legend', 'value': True}, {'label': 'no legend', 'value': False}],
                value=True,
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in ['steps', 'hours']],
                value='steps',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='10000_controller/mean'
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['linear', 'log']],
                value='log',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column2',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='mean_n_planets_in_each_imagination'
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='yaxis-type2',
                options=[{'label': i, 'value': i} for i in ['linear', 'log', 'same']],
                value='linear',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='secondary',
                options=[{'label': i, 'value': i} for i in ['overlay', 'subplot']],
                value='subplot',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column3',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='manager_mean_ship_p'
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='yaxis-type3',
                options=[{'label': i, 'value': i} for i in ['linear', 'log', 'same']],
                value='linear',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='tertiary',
                options=[{'label': i, 'value': i} for i in ['overlay', 'subplot']],
                value='subplot',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column4',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='manager/mean'
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='yaxis-type4',
                options=[{'label': i, 'value': i} for i in ['linear', 'log', 'same']],
                value='log',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            html.Button('Shuffle colors', id='button')
        ], style={}),
        html.Div([
            dcc.RadioItems(
                id='color-bind',
                options=[{'label': i, 'value': i} for i in ['c_id', 'c_han']],
                value='c_id',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            dcc.RadioItems(
                labelStyle={'display': 'inline-block'},
                id=generate_control_id(setting),
                options=[{'label': str(i), 'value': i} for i in values],
                value=list(values)[0]
            # ) if setting in ['blind', 'game', 'early', 'pon'] else
            ) if setting in ['blind', 'early', 'pon'] else
            dcc.Checklist(
                labelStyle={'display': 'inline-block'},
                id=generate_control_id(setting),
                options=[{'label': str(i), 'value': i} for i in values],
                values=list(values)
            ) for setting, values in runs.conf.items() if setting not in ('v', 'id') and groups[0].conf[setting] != 'grouped'
        ]),
        dash_table.DataTable(
            id='table',
            columns=[{"name": 'c', "id": 'c'}, {"name": 'n', "id": 'n'}] + [{"name": x, "id": x} for x, y in groups[0].conf.items() if y != 'grouped'],
            data=[group.dict(n) for n, group in enumerate(groups)],
            row_selectable=True,
            sorting=True,
            css=[
                {"selector": ".dash-spreadsheet-container .dash-spreadsheet-inner td", "rule": 'height: 5px"'}
            ],
            style_table={'textAlign': 'left', 'height': '5px', 'minWidth': '0px', 'padding': '0', 'margin': '0'},
            style_cell={'textAlign': 'left', 'height': '5px', 'minWidth': '0px', 'overflow': 'hidden', 'padding': '0', 'margin': '0'},
            style_data_conditional=[{"if": {"filter": 'c eq num({})'.format(i), 'column_id': 'c'}, 'backgroundColor': color_string(get_color(i, shuffle=i_color)), 'color': color_string(get_color(i, shuffle=i_color))} for i, _ in enumerate(groups)],
            selected_rows=[0]
        ),

    ], style={'float': 'left', 'height': '100%', 'margin-right': '100px'}),

    dcc.Graph(id='indicator-graphic', animate=False, config=dict(scrollZoom=True, clear_on_unhover=True, showAxisDragHandles=True, showAxisRangeEntryBoxes=True, autoSizable=True, responsive=True, editable=True, toImageButtonOptions=dict(
        format='svg',
    )), style={'float': 'right', 'flex': '1', 'width': '90%', 'height': '90%', 'background-color': 'gray', 'padding': '10px'}),
], style={'display': 'flex', 'height': '98vh'})


@app.callback(
    [
        Output('table', 'data'),
        Output('table', 'selected_rows')
    ],
    [Input(generate_control_id(setting), 'value' if setting in ['blind', 'early', 'pon'] else 'values') for setting in runs.conf if setting not in ('v', 'id') and groups[0].conf[setting] != 'grouped'],
    [State('table', "derived_virtual_data"), State('table', "derived_virtual_selected_rows"), State('xaxis-column', 'value')]
)
def display_controls(*args):
    print(args)
    filter = args[:-3]
    rows, selected_rows, xax = args[-3:]

    filtered_groups = []

    for group in groups:
        member = True

        for setting, filter_values in zip(list(runs.conf), filter):
            if (isinstance(filter_values, list) and group.conf[setting] not in filter_values) or (not isinstance(filter_values, list) and group.conf[setting] != filter_values):
                member = False

        if member:
            filtered_groups.append(group)

    data = [group.dict(n, 'times' if xax == 'hours' else 'steps') for n, group in enumerate(groups) if group in filtered_groups]

    preexisting_group_ids = [row['c'] for row in rows] if rows is not None else []
    selected_group_ids = [rows[i]['c'] for i in selected_rows] if selected_rows is not None else []
    new_selected_rows = [i for i, group in enumerate(filtered_groups) if (groups.index(group) not in preexisting_group_ids) or (groups.index(group) in selected_group_ids)]

    return data, new_selected_rows


def generate_output_id(value1, value2):
    return '{} {} container'.format(value1, value2)


@app.callback(
    dash.dependencies.Output('width-output', 'children'),
    [dash.dependencies.Input('width', 'value')])
def update_output(value):
    return value


@app.callback(
    dash.dependencies.Output('height-output', 'children'),
    [dash.dependencies.Input('height', 'value')])
def update_output(value):
    return value


@app.callback([
    Output('indicator-graphic', 'figure'),
    Output('table', "style_data_conditional"),
], [
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('yaxis-column2', 'value'),
    Input('yaxis-type', 'value'),
    Input('yaxis-type2', 'value'),
    Input('table', "derived_virtual_data"),
    Input('table', "derived_virtual_selected_rows"),
    Input('button', 'n_clicks'),
    Input('color-bind', 'value'),
    Input('secondary', 'value'),
    Input('zoom', 'value'),
    Input('xzoom', 'value'),
    Input('height', 'value'),
    Input('width', 'value'),
    Input('line-width', 'value'),
    Input('shade', 'value'),
    Input('diamonds', 'value'),
    Input('cards', 'value'),
    Input('legend', 'value'),
    Input('yaxis-column3', 'value'),
    Input('yaxis-type3', 'value'),
    Input('tertiary', 'value'),
    Input('yaxis-column4', 'value'),
    Input('yaxis-type4', 'value'),
], [
    State('filter-pon', 'value'),
]
)
def update_graph(xaxis_column_name, yaxis_column_name, yaxis_column_name2, yaxis_type, yaxis_type2, rows, selected_rows, button, color_bind, secondary, zoom, xzoom, h, w, line_width, shade, diamonds, cards, legend, yaxis_column3, yaxis_type3, tertiary, yaxis_column4, yaxis_type4, pon):

    if button is not None:
        global i_color
        i_color = button

    print(rows)
    print(selected_rows)

    traces = []
    annotations = []

    selected_group_ids = [rows[i]['c'] for i in selected_rows] if selected_rows is not None else []

    print('triggered', dash.callback_context.triggered)
    new_xax = 'xaxis-column.value' in [thing['prop_id'] for thing in dash.callback_context.triggered]
    print(new_xax)

    sec = secondary != 'overlay'

    for i, group in enumerate(groups):
        if (selected_rows is not None) and (i in selected_group_ids):
            the_color = i if color_bind == 'c_id' else (list(runs.conf['ent']).index(group.conf['ent']) if 'ent' in runs.conf else list(runs.conf['han']).index(group.conf['han']))
            trace, annotation = group.trace(
                yaxis_column_name,
                get_color(the_color, shuffle=i_color),
                yaxis_column_name2,
                sec,
                xaxis_column_name == 'hours',
                line_width,
                shade,
                diamonds,
                yaxis_column3,
                yaxis_column4
            )

            traces += trace
            annotations += annotation

    xax = go.layout.XAxis(automargin=False, showline=True, rangemode='tozero', range=([0, 24.1 * xzoom] if xaxis_column_name == 'hours' else [0, 60000]),
                          anchor='y4', title='episodes trained')

    traces.append(
        go.Scatter(
            name='base',
            x=groups[0].df['steps'].index,
            y=[0.03579 for _ in groups[0].df['steps'].index],
            line=go.scatter.Line(dash='dot', color='rgba(0, 0, 0, 0.4)'),
            mode='lines',
            showlegend=False,
            yaxis='y1',
            hoverinfo='skip'),
    )
    traces.append(
        go.Scatter(
            name='base',
            x=groups[0].df['steps'].index,
            y=[0.03579 + 1*pon for _ in groups[0].df['steps'].index],
            line=go.scatter.Line(dash='dot', color='rgba(0, 0, 0, 0.4)'),
            mode='lines',
            showlegend=False,
            yaxis='y4',
            hoverinfo='skip'),
    )
    traces.append(
        go.Scatter(
            name='base',
            x=groups[0].df['steps'].index,
            y=[0.03579 + 4*pon for _ in groups[0].df['steps'].index],
            # marker=go.scatter.Marker(color='black', symbol='dash'),
            line=go.scatter.Line(dash='dot', color='rgba(0, 0, 0, 0.4)'),
            mode='lines',
            showlegend=False,
            yaxis='y4',
            hoverinfo='skip'),
    )




    if yaxis_type == 'log':
        yrange = [math.log10(0.02), math.log10(zoom)]
    else:
        yrange = [0, zoom]

    return ({
                'data': traces,
                'layout': go.Layout(
                    height=h,
                    width=w,
                    autosize=False,
                    # yaxis=go.layout.YAxis(type='linear' if yaxis_type == 'Linear' else 'log', hoverformat=".4f", automargin=True, showline=True),
                    yaxis=go.layout.YAxis(type=yaxis_type, automargin=False, range=yrange, showline=True,
                                          domain=([0.625, 1] if yaxis_column3 is not None else [0.25, 1]) if yaxis_column_name2 is not None and sec else [0, 1], title='task loss'),
                    yaxis2=go.layout.YAxis(
                        type=yaxis_type if yaxis_type2 == 'same' else yaxis_type2,
                        automargin=False,
                        showline=False,
                        overlaying='free',
                        side='left',
                        zeroline=True,
                        rangemode='tozero',
                        range=[0,4],
                        domain=[0.510, 0.605],
                        title='objects'
                    ),
                    yaxis3=go.layout.YAxis(
                        type=yaxis_type if yaxis_type3 == 'same' else yaxis_type3,
                        automargin=False,
                        showline=False,
                        overlaying='free',
                        side='left',
                        zeroline=True,
                        rangemode='tozero',
                        range=[0.2, 1],
                        domain=[0.395, 0.490],
                        title='P[ship]'
                    ),
                    yaxis4=go.layout.YAxis(type=yaxis_type4, automargin=False, range=[math.log10(0.03), math.log10(zoom)], showline=True, domain=([0.0, 0.375]),title='management loss'),
                    xaxis=xax,
                    margin={'l': 50, 'b': 60, 't': 10, 'r': 70},
                    showlegend=legend,
                    legend=go.layout.Legend(x=0.5, xanchor='center', y=1, bgcolor='rgba(255,255,255,1)', orientation='h'),
                    dragmode='pan',
                    # clickmode='event+select',
                    hoverlabel=go.layout.Hoverlabel(namelength=-1),
                    uirevision=True,
                    annotations=annotations if cards else []
                )
            },
            [{
                "if": {"filter": 'c eq num({})'.format(i), 'column_id': 'c'},
                'backgroundColor': color_string(get_color(i if color_bind == 'c_id' else (list(runs.conf['ent']).index(group.conf['ent']) if 'ent' in runs.conf else list(runs.conf['han']).index(group.conf['han'])), shuffle=i_color)),
                'color': color_string(get_color(i if color_bind == 'c_id' else (list(runs.conf['ent']).index(group.conf['ent']) if 'ent' in runs.conf else list(runs.conf['han']).index(group.conf['han'])), shuffle=i_color))
            } for i, group in enumerate(groups)]
    )


# https://github.com/plotly/dash-table

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
