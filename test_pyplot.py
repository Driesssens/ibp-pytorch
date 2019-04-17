import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

from utilities import get_color, color_string
import pandas as pd
import plotly.graph_objs as go

from pathlib import Path
from cool_data_analysis import Runs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = Path() / 'storage' / 'final' / 'formal1'
# runs = Runs(path, done_hours=20, done_steps=100000).only_done()
runs = Runs(path, done_hours=20, done_steps=100000).only_done()
groups = runs.group(['v'])

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
                value='controller/mean'
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
            )], style={}),
        html.Div([
            dcc.RadioItems(
                id='yaxis-type2',
                options=[{'label': i, 'value': i} for i in ['linear', 'log', 'same']],
                value='same',
                labelStyle={'display': 'inline-block'}
            )], style={}),
        html.Div([
            html.Button('Shuffle colors', id='button')
        ], style={}),
        html.Div([
            dcc.Checklist(
                labelStyle={'display': 'inline-block'},
                id=generate_control_id(setting),
                options=[{'label': str(i), 'value': i} for i in values],
                values=list(values)
            ) for setting, values in runs.conf.items() if setting not in ('v', 'id')
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
            selected_rows=[]
        ),
        # html.Div([
        #     dcc.Checklist(
        #         id='selected-runs',
        #         options=[{'label': i, 'value': i} for i in available_group_names],
        #         values=available_group_names
        #     )], style={}),

    ], style={'float': 'left', 'height': '100%'}),

    dcc.Graph(id='indicator-graphic', animate=True, config=dict(scrollZoom=True, showAxisDragHandles=True, showAxisRangeEntryBoxes=True, autoSizable=True, responsive=True), style={'float': 'right', 'flex': '1', 'width': '90%', 'height': '90%'}),
], style={'display': 'flex', 'height': '98vh'})


@app.callback(
    [
        Output('table', 'data'),
        Output('table', 'selected_rows')
    ],
    [Input(generate_control_id(setting), 'values') for setting in runs.conf if setting not in ('v', 'id')],
    [State('table', "derived_virtual_data"), State('table', "derived_virtual_selected_rows"), ]
)
def display_controls(*args):
    filter = args[:-2]
    rows, selected_rows = args[-2:]

    filtered_groups = []

    for group in groups:
        member = True

        for setting, filter_values in zip(list(runs.conf), filter):
            if group.conf[setting] not in filter_values:
                member = False

        if member:
            filtered_groups.append(group)

    data = [group.dict(n) for n, group in enumerate(groups) if group in filtered_groups]

    preexisting_group_ids = [row['c'] for row in rows]
    selected_group_ids = [rows[i]['c'] for i in selected_rows]
    new_selected_rows = [i for i, group in enumerate(filtered_groups) if (groups.index(group) not in preexisting_group_ids) or (groups.index(group) in selected_group_ids)]

    return data, new_selected_rows


def generate_output_id(value1, value2):
    return '{} {} container'.format(value1, value2)


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
]
)
def update_graph(xaxis_column_name, yaxis_column_name, yaxis_column_name2, yaxis_type, yaxis_type2, rows, selected_rows, button):
    if button is not None:
        global i_color
        i_color = button

    print(rows)
    print(selected_rows)

    traces = []

    selected_group_ids = [rows[i]['c'] for i in selected_rows]

    for i, group in enumerate(groups):
        if (selected_rows is not None) and (i in selected_group_ids):
            traces += group.trace(yaxis_column_name, get_color(i, shuffle=i_color), yaxis_column_name2, yaxis_type2 == 'same')

    return ({
                'data': traces,
                'layout': go.Layout(
                    autosize=True,
                    # yaxis=go.layout.YAxis(type='linear' if yaxis_type == 'Linear' else 'log', hoverformat=".4f", automargin=True, showline=True),
                    yaxis=go.layout.YAxis(type=yaxis_type, automargin=True, showline=True),
                    yaxis2=go.layout.YAxis(
                        type=yaxis_type if yaxis_type2 == 'same' else yaxis_type2,
                        automargin=True,
                        showline=True,
                        side='right',
                        overlaying='y',
                        zeroline=False
                    ) if yaxis_column_name2 is not None and yaxis_type2 != 'same' else None,
                    xaxis=go.layout.XAxis(automargin=True, showline=True),
                    margin={'l': 40, 'b': 25, 't': 25, 'r': 25},
                    showlegend=False,
                    dragmode='pan',
                    hoverlabel=go.layout.Hoverlabel(namelength=-1)
                )
            },
            [{"if": {"filter": 'c eq num({})'.format(i), 'column_id': 'c'}, 'backgroundColor': color_string(get_color(i, shuffle=i_color)), 'color': color_string(get_color(i, shuffle=i_color))} for i, _ in enumerate(groups)]
    )


# https://github.com/plotly/dash-table

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
