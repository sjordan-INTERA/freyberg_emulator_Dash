"""
Dash application for Freyberg Model Emulator

@author: spencerjordan
"""
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_daq as dq
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import timedelta
import dash_vtk
from dash_vtk.utils import to_mesh_state
import pyvista as pv
import sys
import os
import pyemu


global forecast_results
forecast_results = pd.DataFrame()

# Define the app
app = Dash(__name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            )
server = app.server  # --> This is for Docker

# Copying over from workflow.py to avoid import
def response_matrix_emulator(dv_pars,
                             forecast_df,
                             md="assets/master",
                             response_jcb="response.jcb",
                             ):
    """
    
    Calculate the forecast response given changes in decision variables
    
    Parameters
    ----------
    dv_pars : pd.DataFrame
        DataFrame with 'parval1' column and index containing dv_par parameter names
    md : str, optional
        model directory, by default "master"
    response_jcb : str or pyemu.matrix, optional
        response matrix file, by default "response.jcb"
    forecast_csv : str, optional
        forecast response file, by default "forecast_response.csv"

    Returns
    -------
    pd.DataFrame
        DataFrame with forecast response, change and original values.
        
    """

    # dv_pars must be a DataFrame with a 'parval1' column 
    assert isinstance(dv_pars,pd.DataFrame)
    assert "parval1" in dv_pars.columns, "parval1 column not found in dv_pars"

    pst = pyemu.Pst(os.path.join(md,"freyberg.pst"))
    par = pst.parameter_data
    # check that all of dv_pars index values are in par
    assert set(dv_pars.index).issubset(par.index), "dv_pars index values not found in parameter data"
    
    # get the forecast_names
    forecast_names = forecast_df.index.tolist()

    # get the resp mat
    if isinstance(response_jcb,str):
        assert os.path.exists(os.path.join(md,response_jcb)), "response matrix not found"
        resp_mat = pyemu.Matrix.from_binary(os.path.join(md,response_jcb))
    elif isinstance(response_jcb,pyemu.Matrix):
        resp_mat = response_jcb
    else:
        raise ValueError("response_jcb must be a str or pyemu.Jco")
    # keep only pars in dv_pars...and align
    assert set(dv_pars.index).issubset(resp_mat.col_names), f"dv_pars index values not found in {response_jcb} col_names"
    assert set(forecast_names).issubset(resp_mat.row_names), f"forecast names not found in {response_jcb} row_names"

    # keep only relevant rows and cols
    resp_mat = resp_mat.get(row_names=forecast_names,
                            col_names=dv_pars.index.tolist())
    # align
    dv_pars = dv_pars.loc[resp_mat.col_names, :]
    forecast_df = forecast_df.loc[resp_mat.row_names, :]
    assert dv_pars.shape[0] == resp_mat.shape[1], "dv_pars and response matrix are not aligned"
    assert forecast_df.shape[0] == resp_mat.shape[0], "forecast_df and response matrix are not aligned"

    # calc change in dvpars
    dv_pars["change"] = dv_pars.parval1 - par.loc[dv_pars.index.tolist()].parval1
    #check for nans
    assert dv_pars.change.isnull().any()==False, f"nan values found in dv_pars: {dv_pars.isnull().any()}"

    # mat-vec mult
    resp_vec = resp_mat * dv_pars.change.values

    # calculate the forecast response to changes
    forecast_df["change"] = resp_vec.x.flatten()
    forecast_df["forecast"] = resp_vec.x.flatten() + forecast_df.modelled.values
    # forecast_df.to_csv(os.path.join(md,"forecast_response.csv"))
    return forecast_df


# Get the base path depending on whether the app is frozen or not
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # Path to the temporary folder used by PyInstaller
else:
    base_path = os.path.dirname(os.path.abspath(__name__))
    
# Load data necessary for emulator 
md = os.path.join(base_path, "assets")
pst = pyemu.Pst(os.path.join(md, "master", "freyberg.pst"))
par = pst.parameter_data
obs = pst.observation_data
rm = pyemu.Matrix.from_binary(os.path.join(md,"master","response.jcb"))


# --------------------------------------------------------
# Function to run the emulator with selected perturbations
# --------------------------------------------------------
def run_emulator(input_dict):
    # Grab a copy...
    forecast_df = pst.res.loc[:,["modelled"]].copy()
    
    # Load a copy of the obs data, holding onto all observations
    _df = obs.copy()
    _df.time = _df.time.astype(float)
    _df.sort_values("time",inplace=True)
    
    # This will be the forecasted dataframe
    # We will 'disturb' by each mult-param combo
    forecast_df = forecast_df.loc[_df.obsnme.tolist(),:]
    dv_pars = pst.parameter_data.loc[rm.col_names,:].copy()
    dv_pars["parval1_org"] = dv_pars.parval1.copy() # keep a backup of the original values
    
    # Pumping rate inputs
    if 'welrate' in input_dict.keys():
        for entry in input_dict['welrate']:
            well = entry[0]
            if well == 'all':
                sp1 = entry[1]
                sp2 = entry[2]
                sp_range = [x-1 for x in range(sp1,sp2+1)]
                mult = float(entry[3])
                for well in dv_pars.loc[dv_pars.pargp=='welrate'].index:
                    if int(well.split(':')[2].split('_')[0]) in sp_range:
                        dv_pars.loc[dv_pars.parnme==well,"parval1"] *= mult
            else:
                well = well.split('_')[1]
                sp1 = entry[1]
                sp2 = entry[2]
                mult = float(entry[3])
                for sp in range(sp1,sp2+1):
                    well_key = f'pname:wel_0_inst:{sp-1}_ptype:gr_usecol:3_pstyle:d_idx0:2_idx1:{well}'
                    dv_pars.loc[dv_pars.parnme==well_key,"parval1"] *= mult
  
    # SFR/WTP Inputs  
    if 'sfrinflow' in input_dict.keys():
        for entry in input_dict['sfrinflow']:
            mult = float(entry[3])
            if entry == 'sfr_inflow':
                key = 'pname:sfr_0_inst:0_ptype:gr_usecol:2_pstyle:d_idx0:0'
            else:
                key = 'pname:sfr_0_inst:0_ptype:gr_usecol:2_pstyle:d_idx0:60'
            dv_pars.loc[dv_pars.parnme==key,"parval1"] *= mult
    
    # RCH inputs
    if 'rch' in input_dict.keys():
        for entry in input_dict['rch']:
            mult = float(entry[3])
            key = 'pname:rch_0_inst:0_ptype:cn_pstyle:d'
            dv_pars.loc[dv_pars.parnme==key,"parval1"] *= mult
    
    # Create forecasted results
    forecast_results = response_matrix_emulator(dv_pars, 
                                                forecast_df=forecast_df,
                                                md=os.path.join(md,"master"),
                                                response_jcb=rm,
                                                )

    return forecast_results

# --------------------------------
# Create Homepage Figure using vtk
# --------------------------------
scaled_mesh_botm = pv.read(os.path.join(md,'dis_formatted.vtk'))
mesh_state_botm = to_mesh_state(scaled_mesh_botm,'botm')


# ----------------------------------------------------------------------- #
# -------------------------- Main HTML Layout --------------------------- #
# ----------------------------------------------------------------------- #
app.layout = html.Div([
    # Putting the entire page within a dcc.Loading
    dcc.Loading([
        dcc.Store(id='emulator_inputs',
                  data={}),
        # Header with Title and Logo
        html.Div(children=[
                html.H2("Freyberg Model Emulator",
                        style={'textAlign': 'center',
                               'margin': '0',  # Ensures no default margin
                               'flex': '1'}
                        ),
                html.Div([
                    html.A(href='https://www.intera.com/',
                           target='_blank',
                           children=[
                               html.Img(src='/assets/intera_logo.png',
                                        style={'height': '60px'})
                           ]
                           ),
                ],
                    style={'position': 'absolute',  # Position the logo in the upper-right corner
                           'top': '10px',  # Adjust top spacing as needed
                           'right': '20px',  # Adjust right spacing as needed
                           'z-index': 10  # Ensure logo stays above other content
                    }),
            ],
                style={
                    'display': 'flex',
                    'justify-content': 'center',  # Centers the header text horizontally
                    'align-items': 'center',  # Centers the content vertically
                    'width': '100%',
                    'height': '75px',  # Adjust the height if needed
                    'position': 'relative',  # Ensure the logo is positioned relative to the header
                    'background-color': '#DEE6F8',
                    'padding': '10px 20px',
                }
        ),
        # ----------------------------------------------
        # Three tabs - Intro, instructions, and emulator
        # ----------------------------------------------
        dcc.Tabs([
            dcc.Tab(label='About the Model',
                    children=[
                       html.Div([
                            html.H3('Freyberg Model', style={'text-align': 'center', 'color': '#333', 'font-weight': 'bold'}),
                            html.Hr(style={'border-color': '#ccc'}),
                            
                            # Information Section
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Model Details", style={'color': '#0066cc', 'font-weight': 'bold'}),
                                        html.P(["The Freyberg model is a synthetic groundwater model with 3 layers, 120 rows, and 60 columns. It was originally developed by ",
                                                html.A("David Freyberg", href="https://ngwa.onlinelibrary.wiley.com/doi/10.1111/j.1745-6584.1988.tb00399.x", target="_blank", style={'color': '#007bff', 'text-decoration': 'underline'}),
                                                " as a class exercise in model calibration."
                                                ], 
                                               style={'line-height': '1.6', 'text-align': 'justify'}
                                        ),
                                        html.P("A General Head Boundary (GHB) is located along the southern boundary, while all other external boundaries are set to no-flow.",
                                               style={'line-height': '1.6', 'text-align': 'justify'}),
                                        
                                        html.P("The surface-water system consists of a straight stream flowing north to south, simulated with the Streamflow Routing (SFR) package. The SFR reaches traverse the model domain from row 1 to row 120 in column 48, with surface-water flow observations monitored in reach 120 (the terminal reach).",
                                               style={'line-height': '1.6', 'text-align': 'justify'}),
                                        
                                        html.P("There are six groundwater extraction wells (shown as red cells) and several monitoring wells (shown as green cells).",
                                               style={'line-height': '1.6', 'text-align': 'justify'}),
                                        
                                        html.P("Water enters the model domain through recharge and stream leakage in layer 1, and exits via groundwater discharge to surface water, groundwater extraction, and the downgradient GHB.",
                                               style={'line-height': '1.6', 'text-align': 'justify'}),
                                        html.P("In the figure below, the Z direction is exaggerated 50x, and each cell is colored by its bottom elevation. The SFR stream (purple line) flows from the positive Y direction to the negative Y direction (activate 'Show model axes' to view this). In the numerical model, all pumping wells are within layer 3 and all observations wells are in layer 1.",
                                               style={'line-height': '1.6', 'text-align': 'justify'}),
                                    ], width=12)
                                ]),
                            ], style={
                                'padding': '20px',
                                'background-color': '#f9f9f9',
                                'border': '1px solid #ddd',
                                'border-radius': '8px',
                                'box-shadow': '2px 2px 10px rgba(0,0,0,0.05)',
                                'margin-top': '10px'
                            }),
                        ], style={'width': '95%', 'margin': 'auto', 'padding': '20px', 'background-color': '#ffffff'}
                        ),

                        html.Br(),
                        html.Div([
                            html.H3('Model Grid', style={'text-align': 'center', 'color': '#333', 'font-weight': 'bold'}),
                            # Options for the VTK figure
                            html.Div([
                                html.P('Change grid opacity:'),
                                dcc.Dropdown(placeholder='Grid cell opacity',
                                             id='vtk_opactiy_dd',
                                             options=[x/10 for x in range(1,11)],
                                             value=1,
                                             clearable=False,
                                             style={'margin-right': '10px', 'flex':'1','max-width': '200px'}
                                ),
                                dcc.Checklist(options=[{'label': 'Show grid-cell edges', 'value': 'edges'}],
                                              id='edge-mode-checkbox',
                                              value=[],
                                ),
                                dcc.Checklist(options=[{'label': 'Show model axes', 'value': 'axes'}],
                                              id='axes-checkbox',
                                              value=[],
                                ),
                                ],
                                style={'display':'flex',
                                       'flex-direction':'row',
                                       'align-items': 'center',
                                       'gap': '10px',
                                       'background-color': '#f9f9f9',
                                       'border': '1px solid #ddd',
                                       'border-radius': '8px',
                                       'box-shadow': '2px 2px 10px rgba(0,0,0,0.05)',
                                       'padding': '10px',
                                       }
                            ),
                            html.Hr(),
                            dash_vtk.View([
                                dash_vtk.GeometryRepresentation(
                                    id='vtk-geometry-representation',
                                    children=[
                                        dash_vtk.Mesh(state=mesh_state_botm)
                                    ],
                                    colorMapPreset="erdc_rainbow_bright",  # Use a preset color map
                                    # Set the color range for 'top' values
                                    colorDataRange=[-0.20274, 42.1417],
                                    property={"edgeVisibility": False,
                                              "opacity": 1.0,
                                              },

                                ),
                                ],
                                style={"height": "700px"},
                                background=[0.788, 0.843, 0.969],
                            ),
                            ],
                            style={'padding': '20px',
                                   'background-color': '#f9f9f9',
                                   'border': '1px solid #ddd',
                                   'border-radius': '8px',
                                   'box-shadow': '2px 2px 10px rgba(0,0,0,0.05)',
                                   'margin-top': '10px',
                                   'margin-bottom':'300px',
                                   'width':'95%',
                                   'margin': 'auto'}
                        ),
                    ],
                    style={'fontSize': '120%'},
                    selected_style={'fontSize': '125%'},
            ),
            dcc.Tab(label='Emulator Instructions',
                    children=[
                       html.Div([
                            # Section: Steps to Generate Response Matrix
                            # html.Div([
                            #     html.H4('Steps to Generate the Freyberg Model Response Matrix using PEST', style={'text-align': 'center', 'color': '#333', 'font-weight': 'bold','text-decoration': 'underline'}),
                            #     html.Br(),
                            #     html.P(
                            #         "Before running the Emulator, you must execute 'workflow.py'. Ensure the 'num_workers' parameter within this script is set to match your system's specifications (i.e., the number of available computing cores).",
                            #         style={'line-height': '1.6', 'text-align': 'justify'}
                            #     ),
                            #     html.P(
                            #         "This script will set up and run the Freyberg groundwater model, PstFrom, and pestpp-glm to generate a response matrix. Once this is done, the Dashboard will have access to the response matrix and all necessary output files in order to emulate the groundwater model.",
                            #         style={'line-height': '1.6', 'text-align': 'justify'}
                            #     ),
                            # ], style={
                            #     'padding': '20px',
                            #     'background-color': '#f9f9f9',
                            #     'border': '1px solid #ddd',
                            #     'border-radius': '8px',
                            #     'box-shadow': '2px 2px 10px rgba(0,0,0,0.05)',
                            #     'margin-bottom': '20px'
                            # }),
                        
                            # Section: Using the Emulator
                            html.Div([
                                html.H4('Using the Emulator', style={'text-align': 'center', 'color': '#333', 'font-weight': 'bold','text-decoration': 'underline'}),
                                html.Br(),
                        
                                # Step 1
                                html.H4('1. Select a Parameter Group to Adjust', style={'color': '#0066cc', 'font-weight': 'bold'}),
                                html.P("Each parameter option represents a catagory of decision variables:", style={'line-height': '1.6'}),
                                html.Ul(children=[
                                    html.Li("Pumping Rates: Pumping rates at each extraction well"),
                                    html.Li("Stream and WTP Inflows: inflows from upstream and mid-stream inflows from a water treatment plant"),
                                    html.Li("Antecedent Recharge: recharge in the steady state stress period (stress period 0)")
                                ], style={'line-height': '1.6'}),
                                html.P("Once selected, this will populate a dropdown menu with model parameters that can be adjusted.", style={'line-height': '1.6'}),
                                html.Hr(),
                        
                                # Step 2
                                html.H4('2. Configure Emulator Inputs', style={'color': '#0066cc', 'font-weight': 'bold'}),
                                # Choose a Decision Variable, Specify Desired Time-Period to Adjust, and Input a Multiplier to Apply to the Variable ACross the Specified Stress Periods
                                html.P('Choose a decision parameter from the drowndown menu and a range of stress periods to adjust.',
                                       style={'line-height': '1.2'}),
                                html.P("Enter any multiplier value to adjust the selected decision variable. For instance, entering '2' with 'All Wells' selected will double the pumping rate for all wells during the specifed stress periods.", 
                                       style={'line-height': '1.2'}),
                                html.P("Click the 'save input' button to save the current parameter adjustment to the emulator scenario. You will see this input populate the table to the right.", 
                                       style={'line-height': '1.2'}),
                                html.Ul(children=[
                                    html.Li("Any combination of pumping, SFR, and Recharge parameters can be specifed to create unique scenarios"),
                                    html.Li("An unlimited number of parameter inputs can be specifed and added to the scenario"),
                                    html.Li("To reset the scenario, click the 'Reset Inputs' button. This will remove all inputs to the current emulator scenario"),
                                    ],style={'line-height': '1.6'}
                                ),
                                html.Hr(),
                    
                                # Step 3
                                html.H4('3. Click "Run Emulator" to Produce Results', style={'color': '#0066cc', 'font-weight': 'bold'}),
                                html.P("Once the emulator completes its analysis, navigate to the 'Results viewer' tab to view graphical results", style={'line-height': '1.6'}),
                            ], style={
                                'padding': '20px',
                                'background-color': '#f9f9f9',
                                'border': '1px solid #ddd',
                                'border-radius': '8px',
                                'box-shadow': '2px 2px 10px rgba(0,0,0,0.05)',
                                'margin-top': '20px'
                            }),
                        ], 
                        style={'width': '95%', 'margin': 'auto', 'padding': '20px', 'background-color': '#ffffff','margin-bottom':'100px'}
                    ),
                    ],
                    style={'fontSize': '120%'},
                    selected_style={'fontSize': '125%'},
                    ),
            dcc.Tab(label='Model Emulator',
                    children=[
                        # Main content split between input section and graph
                        html.Div([
                            # Input section
                            html.Div([
                                # View for making the emulator inputs
                                html.Div(
                                    id='input_options_container',
                                    children=[
                                        html.H5("1. Select Paramater Group:"),
                                        # ----------------------------
                                        # Selector for input parameter
                                        # ----------------------------
                                        html.Div([
                                            dcc.RadioItems([{'label': 'Pumping Rates', 'value': 'welrate'},
                                                            {'label': 'Stream and WTP Inflows (All Stress Periods)','value': 'sfrinflow'},
                                                            {'label': 'Antecedent Recharge (1st Stress Period)', 'value': 'rch'}],
                                                           'welrate',
                                                           id='par_options',
                                                           style={'width': '100%',
                                                                  'margin-bottom': '20px',
                                                                  'fontsize': '120%'}
                                                           ),
                                            ],
                                            style={'padding': '5px',
                                                   'border': '2px solid #ccc',
                                                   'border-radius': '5px',
                                                   'background-color': '#f2f2f2',
                                                   'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
                                                   }
                                        ),
                                            
                                        html.Hr(),
                                        
                                        # Input options for each decision parameter
                                        html.H5("2. Options for Selected Parameter Group:"),
                                        
                                        # For Pumping Wells
                                        html.Div(id='pumping_input_options', 
                                                 children=[
                                                    html.Div(id='input_label',
                                                              children=[
                                                                  html.H6('Choose Input Well (layer, row, col):'),
                                                                  ]
                                                    ),
                                                    dcc.Dropdown(
                                                        id='input_well_choice',
                                                        options=[
                                                            {'label': 'All Wells', 'value': 'all'},
                                                            {'label': '2, 28, 49', 'value': '2_28_49'},
                                                            {'label': '2, 34, 40', 'value': '2_34_40'},
                                                            {'label': '2, 61, 43', 'value': '2_61_43'},
                                                            {'label': '2, 79, 31', 'value': '2_79_31'},
                                                            {'label': '2, 88, 19', 'value': '2_88_19'},
                                                            {'label': '2, 103, 37', 'value': '2_103_37'}
                                                        ],
                                                        placeholder="Select Parameter...",
                                                        style={
                                                            'margin-bottom': '5px',
                                                            'width': '90%',
                                                            'padding': '4px',
                                                        }
                                                    ),
                                                    html.H6('Choose starting SP:'),
                                                    dcc.Dropdown(
                                                        id='well_SP_input_1',
                                                        options=[{'label': f'SP {x+1}', 'value': x+1} for x in range(25)],
                                                        placeholder="Select SP...",
                                                        style={
                                                            'margin-bottom': '5px',
                                                            'width': '90%',
                                                            'padding': '4px',
                                                        }
                                                    ),
                                                    html.H6('Choose ending SP (must be greater than starting SP):'),
                                                    dcc.Dropdown(
                                                        id='well_SP_input_2',
                                                        options=[{'label': f'SP {x+1}', 'value': x+1} for x in range(25)],
                                                        placeholder="Select SP...",
                                                        style={
                                                            'margin-bottom': '5px',
                                                            'width': '90%',
                                                            'padding': '4px',
                                                        }
                                                    ),
                                                    html.H6('Input Parameter Multiplier:'),
                                                    dcc.Input(
                                                        id="par_mult_pumping",
                                                        type="text",
                                                        placeholder="Enter numeric multiplier...",
                                                        style={
                                                            'margin-bottom': '5px',
                                                            'width': '50%',
                                                            'padding': '4px',
                                                        }
                                                    ),
                                                ],
                                                style={
                                                    'display': 'flex',
                                                    'flex-direction': 'column',
                                                    'padding': '5px',
                                                    'border': '2px solid #ccc',
                                                    'border-radius': '5px',
                                                    'background-color': '#f2f2f2',
                                                    'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                                                    'width': '100%',
                                                }
                                        ),
                                        # Save input button
                                        dbc.ButtonGroup(
                                            children=[
                                                # Button to save current input
                                                dbc.Button('Save Input',
                                                           id='save_input_button',
                                                           color='success',
                                                           className="d-grid gap-2",
                                                           style={'margin-top':'20px',
                                                                  'width':'100%'}
                                                           ),
                                                # Button to reset inputs
                                                dbc.Button('Reset Inputs',
                                                           id='reset_input_button',
                                                           color='warning',
                                                           className="d-grid gap-2",
                                                           style={'margin-top':'20px',
                                                                  'width':'100%'}
                                                           ),
                                                ],
                                            style={'width':'100%'}
                                        ),
                                        # Modal to warn that input was not saved
                                        dbc.Modal(
                                            id='input_error_modal',
                                            children=[
                                                dbc.ModalHeader("Specified input not saved!"),
                                                dbc.ModalBody(
                                                    "Make sure the inputs are properly formatted.",
                                                    className="align-self-center",
                                                ),
                                            ]
                                        ),
                                       # html.Hr(),
                                        
                                        # Modal that holds the real reset button
                                        dbc.Modal(
                                            id='input_reset_modal',
                                            children=[
                                                dbc.ModalHeader("Are you sure you want to reset all inputs?"),
                                                dbc.ModalBody(
                                                    dbc.Button('Yes, reset.',
                                                               id='input_reset_button_check',
                                                               color='danger'
                                                    ),
                                                    className="align-self-center",
                                                ),
                                                dbc.ModalFooter(
                                                    dbc.Button('Cancel',
                                                               id='cancel_input_reset_button'
                                                    )
                                                )
                                            ]
                                        ),
                                        html.Hr(),
                                        #-------------------------------
                                        # Button to run the "simulation"
                                        # ------------------------------
                                        html.H5("3. Run Emulator with Selected Options:"),
                                        dbc.Button("Run Emulator",
                                                   id='emulate_button',
                                                   color='primary',
                                                   className="d-grid gap-2",
                                                   style={'margin-top':'20px',
                                                          'width':'100%'}
                                                   ),
                                        html.Hr(),
                                    ],
                                ),
                                #----------------------------------
                                # Container for Graph/result viewer
                                # ---------------------------------
                                html.Div(
                                    id='graph_option_container',
                                    children=[
                                        # Selector for response variable
                                        html.H5("Select Response Variable:"),
                                        html.Div([
                                            dcc.RadioItems([{'label': 'Simulated Heads (Contour)',
                                                             'value': 'sim_hds'},
                                                            {'label': 'Streamflow', 'value': 'gage'},
                                                            # {'label': 'SFR Contour', 'value': 'SFR_contour'},
                                                            {'label': 'GW/SW Exchange: Upstream',
                                                             'value': 'headwater'},
                                                            {'label': 'GW/SW Exchange: Downstream',
                                                             'value': 'tailwater'},
                                                            {'label': 'Monitoring Well Heads (Time Series)', 'value': 'monitoring_well_hds'},
                                                            {'label': 'Zone Budget', 'value': 'zone_bud'},],
                                                           'sim_hds',
                                                           id='response_selector',
                                                           style={'width': '100%',
                                                                  'margin-bottom': '20px'}
                                                           ),
                                            ],
                                            style={'padding': '5px',
                                                   'border': '2px solid #ccc',
                                                   'border-radius': '5px',
                                                   'background-color': '#f2f2f2',
                                                   'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                                                   'width': '100%',}
                                        ),
                                        html.Hr(),
    
                                        # --------------------------------------------------
                                        # Graph options --> Will depend on selected variable
                                        # --------------------------------------------------
                                        # For the line graphs
                                        html.Div(id='line_graph_options',
                                                 children=[html.H5('Additional Figure Options',
                                                                   style={'text-align': 'center'}),
                                                           html.Hr(),
                                                           html.Div('Slide to Show Cumulative Values',
                                                                    style={'text-align': 'center'}),
                                                           dq.BooleanSwitch(
                                                     id='cum_switch')
                                                 ],
                                                 style={'display': 'none'}
                                                 ),
                                        # For the contour plots
                                        html.Div(id='contour_graph_options',
                                                 children=[html.H5('Additional Figure Options',
                                                                   style={'text-align': 'center'}),
                                                           html.Hr(),
                                                           html.Div('Slide to Select Stress Period',
                                                                    style={'text-align': 'center'}),
                                                           dcc.Slider(1, 24,
                                                                      step=1,
                                                                      id='SP_slider',
                                                                      value=1
                                                                      ),
                                                           html.Hr(),
                                                           html.Div('Select Model Layer',
                                                                    style={'text-align': 'left'}),
                                                           dcc.RadioItems([{'label': 'Layer 1', 'value': 0},
                                                                           {'label': 'Layer 3', 'value': 2}],
                                                                          id='layer_selector',
                                                                          value=0
                                                                          ),
                                                           html.Hr(),
                                                           html.Div('Click on the red monitoring wells to display head time-series',
                                                                    style={'text-align': 'left'}),
                                                 ],
                                                 style={'display': 'none'}
                                                 ),
                                        # For the Sankey diagram
                                        html.Div(id='sanky_options',
                                                 children=[html.H5('Additional Figure Options',
                                                                   style={'text-align': 'center'}),
                                                           html.Hr(),
                                                           html.Div('Select Figure Type',
                                                                    style={'text-align': 'left'}),
                                                           dcc.RadioItems([{'label': 'Flow Diagram', 'value': 'flow_diagram'},
                                                                           {'label': 'Time Series', 'value': 'time_series'}],
                                                                          id='zonbud_selector',
                                                                          value='flow_diagram'
                                                                          ),
                                                           html.Hr(),
                                                           html.Div(
                                                               id='zb_sankey_section',
                                                               children=[
                                                                   html.H6('Slide to Select Stress Period',
                                                                           style={'text-align': 'center'}),
                                                                   dcc.Slider(1, 24,
                                                                              step=1,
                                                                              id='SP_slider_sanky',
                                                                              value=1
                                                                              ),
                                                                   html.Hr(),
                                                                   ]
                                                           ),
                                                           html.Div(
                                                               id='zb_ts_section',
                                                               children=[
                                                                   html.H6('Select Aquifer and Zone-Budget Variable',
                                                                           style={'text-align': 'center'}),
                                                                   # dropdown for zonebud component selection
                                                                   dcc.Dropdown(id='zonbud_ts_aquifer',
                                                                                options=[{'label': 'Aquifer 1', 'value': '1'},
                                                                                         {'label': 'Aquifer 2',
                                                                                          'value': '2'},
                                                                                         {'label': 'Aquifer 3',
                                                                                          'value': '3'}
                                                                                         ],
                                                                                value='1'
                                                                                ),
                                                                   dcc.Dropdown(id='zonbud_dropdown',
                                                                                options=[{'label': "From aquifer 1", 'value': 'from-zone-1'},
                                                                                         {'label': "From aquifer 2",
                                                                                          'value': 'from-zone-2'},
                                                                                         {'label': "From aquifer 3",
                                                                                          'value': 'from-zone-3'},
                                                                                         {'label': "To aquifer 1",
                                                                                          'value': 'to-zone-1'},
                                                                                         {'label': "To aquifer 2",
                                                                                          'value': 'to-zone-2'},
                                                                                         {'label': "To aquifer 3",
                                                                                          'value': 'to-zone-3'},
                                                                                         {'label': "From river",
                                                                                          'value': 'sfr-in'},
                                                                                         {'label': "To river",
                                                                                          'value': 'sfr-out'},
                                                                                         {'label': "From tributary",
                                                                                          'value': 'ghb-in'},
                                                                                         {'label': "To tributary",
                                                                                          'value': 'ghb-out'},
                                                                                         ],
                                                                                value='sfr-out'
                                                                                ),
                                                                   html.Hr()
                                                                   ]
                                                           ),
                                                       ],
                                                 style={'display': 'none'}
                                                 ),
                                        ]
                                    ),
                            ], style={
                                'width': '25%',
                                'display': 'inline-block',
                                'padding': '20px',
                                'border': '2px solid #ccc',  # Box border
                                'border-radius': '5px',  # Rounded corners
                                'background-color': '#f9f9f9',  # Light background color
                                # Subtle shadow
                                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
                            }
                            ),
                                
                            # -------------
                            # Graph section
                            # -------------
                            html.Div([
                               
                                # Two tabs: One for inputs, one for figures
                                dcc.Tabs(
                                    id='emulator_tab',
                                    children=[
                                        dcc.Tab(
                                            label='Emulator Configuration',
                                            children=[
                                                html.Div(id='input_container'),
                                                html.Div(id='success_container',
                                                         style={'display':'none'}),
                                            ],
                                        ),
                                        dcc.Tab(
                                            label='Results Viewer',
                                            children=[
                                                dcc.Graph(id='graph-output',
                                                          style={'height': '80vh',
                                                                 }
                                                )
                                            ]
                                        ),
                                    ]
                                )
                                ],
                                style={'width': '75%',
                                       'display':'inline-block',
                                       'padding': '10px',
                                       'border': '2px solid #ccc',  # Box border
                                       'border-radius': '5px',  # Rounded corners
                                       'background-color': '#f9f9f9',  # Light background color
                                       # Subtle shadow
                                       'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'
                                 }
                            ),
                        ], 
                        style={'display': 'flex', 'flex-direction': 'row'}
                        ),
                        dcc.Loading(children=[
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(
                                        dbc.ModalTitle("Well Timeseries")),
                                    dbc.ModalBody("Temp", id='modal_body'),
                                    dbc.ModalFooter(),
                                    dcc.Graph(id='modal_graph')
                                ],
                                size='lg',
                                id="modal",
                                is_open=False,
                            ),

                        ]
                        ),
                    ],
                    style={'fontSize': '120%'},
                    selected_style={'fontSize': '125%'},
                    ),
        ],
        )],
                                
        # Loading spinner options
        target_components={"success_container": "children"},
        overlay_style={"visibility": "visible", "filter": "blur(2px)"},
        custom_spinner=html.H2(
            ["Loading Emulator Results...", dbc.Spinner(color="primary")])
    ),
])


# ----------------------------                            
# Update vtk figure components
# ----------------------------                          
@app.callback(Output('vtk-geometry-representation','property'),
              Output('vtk-geometry-representation','showCubeAxes'),
              Input('vtk_opactiy_dd','value'),
              Input('edge-mode-checkbox','value'),
              Input('axes-checkbox','value'),
              prevent_initial_call=True
              )                                
def update_vtk_fig(opacity,edges,axes):
    # Edge mode
    if edges == ['edges']:
        edge = True
    else:
        edge = False
        
    # Show axes?
    if axes == ['axes']:
        axe = True
    else:
        axe = False    
    # Opacity value
    opacity = float(opacity)
    
    return {"edgeVisibility": edge,"opacity":opacity}, axe

# -----------------------                             
# Update the inputs table      
# -----------------------                         
@app.callback(Output('input_container','children'),
              Input('emulator_inputs','data'),
              )
def update_input_table(data):
    # Create table header
    table_header = html.Thead(html.Tr([
        html.Th("Parameter Group"),
        html.Th("Subgroup"),
        html.Th("Start SP"),
        html.Th("End SP"),
        html.Th("Multiplier")
    ]))
            
    # Create table rows based on the dictionary data
    table_rows = []
    for group, entries in data.items():
        for entry in entries:
            # Some table formatting
            if group == 'sfrinflow':
                end_sp = 24
            else:
                end_sp = entry[2]
            table_rows.append(
                html.Tr([
                    html.Td(group),             # Parameter Group
                    html.Td(entry[0]),          # Subgroup
                    html.Td(entry[1]),          # Start SP
                    html.Td(end_sp),          # End SP
                    html.Td(entry[3])           # Multiplier
                ])
            )
    
    if len(data.keys()) > 0:
        # Define the complete table
        table = dbc.Table(
            [table_header] + [html.Tbody(table_rows)],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            style={'text-align': 'center', 'width': '100%'}
        ) 
    else:
        table = dbc.Table(
            [table_header],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            style={'text-align': 'center', 'width': '100%'}
        ) 
    return table
                     
# ---------------------------------------------
# Saves the specified inputs to the dcc.Store()   
# ---------------------------------------------             
@app.callback(Output('emulator_inputs','data'),
              Output('input_error_modal','is_open'),
              Input('save_input_button','n_clicks'),
              # Existing emulator data
              State('emulator_inputs','data'),
              # Parameter assocaited with input --> 'param'
              # Will be one of: welrate, sfrinflow, rch
              State('par_options','value'),
              State('input_well_choice','value'),
              State('well_SP_input_1','value'),
              State('well_SP_input_2','value'),
              State('par_mult_pumping','value'),
              State('input_error_modal','is_open'),
              prevent_initial_call=True
              )
def update_emulator_inputs(clicked,existing_inputs,param,well,sp1,sp2,mult,modal_open):
    if param == 'welrate':
        # If input incomplete, do not edit inputs
        if not all([well,sp1,sp2,mult]):
            return existing_inputs, not modal_open
        
        # Check to make sure sp1 is less than sp2
        if int(sp1) > int(sp2):
            return existing_inputs, not modal_open
    else:
        sp1 = 1
        sp2 = 1
    try:
        float(mult)
    except:
        return existing_inputs, not modal_open
    
    if param in existing_inputs.keys():
        well_inputs = existing_inputs[param].copy()
        well_inputs.append([well,sp1,sp2,mult])
    else:
        well_inputs = []
        well_inputs.append([well,sp1,sp2,mult])
    existing_inputs[param] = well_inputs
            
    return existing_inputs, modal_open        

# ---------------------------------------------------
# Reset the input fields when input button is clicked
# ---------------------------------------------------
@app.callback(Output('input_well_choice','value'),
              Output('well_SP_input_1','value'),
              Output('well_SP_input_2','value'),
              Output('par_mult_pumping','value'),
              Input('save_input_button','n_clicks'),
              prevent_initital_call=True
              )
def reset_input_form(clicked):
    return None,None,None,None

# --------------------------------------------------
# Callback to ensure user wants to delete all inputs
# --------------------------------------------------
@app.callback(Output('input_reset_modal','is_open'),
              Input('reset_input_button','n_clicks'),
              Input('cancel_input_reset_button','n_clicks'),
              Input('input_reset_button_check','n_clicks'),
              State('input_reset_modal','is_open'),
              prevent_initial_call=True
              )
def toggle_modal_wells(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open

# -----------------------------
# Delete all the current inputs
# -----------------------------
@app.callback(Output('emulator_inputs','data',
                     allow_duplicate=True),
              Input('input_reset_button_check','n_clicks'),
              prevent_initial_call=True
              )
def reset_all_inputs(clicks):
    return {}

# ------------------------------------------------------
# Update the parameter options and label
# This will also deactivate some options for RCH and SFR
# ------------------------------------------------------
@app.callback(Output('input_well_choice','options'),
              Output('input_label','children'),
              Output('well_SP_input_1','disabled'),
              Output('well_SP_input_1','placeholder'),
              Output('well_SP_input_2','disabled'),
              Output('well_SP_input_2','placeholder'),
              Input('par_options','value'),
              )
def update_param_options(par):
    if par == 'welrate':
        options=[
            {'label': 'All Wells', 'value': 'all'},
            {'label': '2, 28, 49', 'value': '2_28_49'},
            {'label': '2, 34, 40', 'value': '2_34_40'},
            {'label': '2, 61, 43', 'value': '2_61_43'},
            {'label': '2, 79, 31', 'value': '2_79_31'},
            {'label': '2, 88, 19', 'value': '2_88_19'},
            {'label': '2, 103, 37', 'value': '2_103_37'}
        ]
        return options, html.H6('Choose Input Well (layer, row, col):'), False, 'Select SP', False, 'Select SP'
    elif par == 'sfrinflow':
        options=[
            {'label': 'Upstream SFR Inflow', 'value': 'sfr_inflow'},
            {'label': 'WTP Inflow', 'value': 'wtp_inflow'},
        ]
        return options, html.H6('Choose SFR Input:'), True, 'Unavailable', True, 'Unavailable'
    elif par == 'rch':
        options=[
            {'label': 'Antecedant Recharge', 'value': 'antecedant_rch'},
        ]
        return options, html.H6('Choose Recharge Input:'),  True, 'Unavailable', True, 'Unavailable'

# --------------------------------------------------
# Update the left-side options based on selected tab
# --------------------------------------------------
@app.callback(Output('input_options_container','style'),
              Output('graph_option_container','style'),
              Input('emulator_tab','value')
              )
def update_option_visibility(tab_state):
    if tab_state == 'tab-1':
        return {},{'display':'none'}
    else:
        return {'display':'none'},{}

# -------------------------------------------
# Update emulator data when button is pressed
# -------------------------------------------
@app.callback(Output('success_container','children'),
              Input('emulate_button','n_clicks'),
              State('emulator_inputs','data'),
              prevent_initial_call=True
              )

def calc_emulator_results(clicks,input_data):
    global forecast_results
    forecast_results = run_emulator(input_data)
    return 'Emulator run success! See "Results Viewer" tab'


# -----------------------------------------------
# Update visibility of Zone Budget result options
# -----------------------------------------------
@app.callback(Output('zb_ts_section','style'),
              Output('zb_sankey_section','style'),
              Input('response_selector','value'),
              Input('zonbud_selector','value')
              )
def zb_graph_options(resp,zb_choice):
    if zb_choice == 'flow_diagram':
        return {'display':'none'},{}
    else:
        return {},{'display':'none'}


# ---------------------------------------------------------------- #
# -------------- Callback function to Plot Results --------------- #
# -------  One big 'ol callback to handle all the plotting ------- #
# ---------------------------------------------------------------- #
@app.callback(Output('graph-output','figure'),
              Output('line_graph_options','style'),
              Output('contour_graph_options','style'),
              Output('sanky_options','style'),
              Input('cum_switch','on'),
              Input('response_selector','value'),
              Input('SP_slider','value'),
              Input('SP_slider_sanky','value'),
              Input('layer_selector','value'),
              Input('zonbud_selector','value'),
              Input('zonbud_dropdown','value'),
              Input('zonbud_ts_aquifer','value'),
              Input('success_container','children'),
              State('emulate_button','n_clicks'),
              prevent_initial_call=True,
              )
def plot_emulator_results(switch,response,SP_slider,SP_slider_sanky,layer_slider,zonbud_option,zonbud_dropdown,zonbud_ts_aquifer,trigger,nclicks): 
    if not nclicks:
        fig = make_subplots()
        return fig,{'display':'block'},{'display':'none'},{'display':'none'}
    else:
        # Assert the results as a global variable
        global forecast_results
        
        # ------------------------------------------------------
        # ---- Line plots: SFR stream, up/down stream conditions
        # ------------------------------------------------------
        if response in ['gage','headwater','tailwater']:
            fig = make_subplots()
            
            # Load the forecasted DataFrame for each specified multiplier
            disturbed_forecast = forecast_results.copy()   
                             
            #disturbed_forecast = disturbed_forecast.set_index('name')
            disturbed_forecast = disturbed_forecast.loc[obs.usecol==response].copy()
            times = [float(i.split("time:")[-1]) for i in disturbed_forecast.index]
            start_date = pd.Timestamp('2022-01-01')
            dates = [start_date + timedelta(days=t) for t in times]
            
            # Plot cumulative sum of values, if specified
            if switch:
                disturbed_forecast[['modelled','forecast']] = disturbed_forecast[['modelled','forecast']].cumsum()
            
            # Plot the original and disturbed timeseries
            fig.add_trace(go.Scatter(x=dates,
                                     y=disturbed_forecast.modelled,
                                     mode='lines',
                                     name='original',
                                     line=dict(color='black',
                                               )
                                     ),
                          )
            
            # Plot disturbed forecast
            fig.add_trace(go.Scatter(x=dates,
                                     y=disturbed_forecast.forecast,
                                     mode='lines',
                                     name='Scenario Results',
                                     #line=dict(color='red'),
                                     line=dict(dash='dash'),
                                     )
                          )
            
            # Add a zero line for the GW/SW interaction plots
            fig.add_hline(y=0.0)
                
            layout = go.Layout(title=f'Observation Timeseries Group: {response}',
                               yaxis=dict(title='Rate (m<sup>3</sup>/day)'),
                               xaxis=dict(title='Date'))
            fig.update_layout(layout)
            # Match the webpage background color
            fig.layout.paper_bgcolor = '#F2F2F2'
            fig.layout.plot_bgcolor = '#DAEDFF'
            fig.update_layout(yaxis=dict(tickformat=".1e"))
            return fig,{'display':'block'},{'display':'none'},{'display':'none'}
        
        
        # --------------------------------
        # ---- Zone Budget - sanky diagram
        # --------------------------------
        elif response == 'zone_bud':
            
            # zonebud sanky diagram
            if zonbud_option=='flow_diagram':
                # Stress period slider value
                SP = SP_slider_sanky
                
                # ZB for layer 1
                _df_1 = obs.loc[
                        (obs.oname == "zbg") & (obs.zone == '1') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                # ZB for layer 2
                _df_2 = obs.loc[
                        (obs.oname == "zbg") & (obs.zone == '2') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                # ZB for layer 3
                _df_3 = obs.loc[
                        (obs.oname == "zbg") & (obs.zone == '3') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                
                # Values for Sankey diagram
                # Indices are hard-referenced
                # If adding new variables, need to rework all inputs to "values" array
                values = []
                # Append data from layer 1
                [values.append(x) for x in _df_1.iloc[[8,9,4,5,6]]['obsval'].values]

                # Append data from layer 2
                [values.append(x) for x in _df_2.iloc[[7,9,4,5,6]]['obsval'].values]

                # Append data from layer 3
                [values.append(x) for x in _df_3.iloc[[7,8,4,5,6]]['obsval'].values]

                # Now calculate net flows
                net_values_1 = []
                # net flow 1->2, 1->3, 2->3, and then aquifer->tributary and aquifer->stream for each layer
                [net_values_1.append(x) for x in [values[0]-values[5],      # 1-->2
                                                  values[1]-values[10],      # 1-->3
                                                  values[6]-values[11],      # 2-->3
                                                  values[2],values[4]-values[3],      # Layer 1
                                                  values[7],values[9]-values[8],      # Layer 2
                                                  values[12],values[14]-values[13]    # Layer 3
                                                  ]
                 ]
                
                # Calculate the flow directions based on flow values
                # There is probably a cleaner way to do this...
                def calc_net_flow_directions(net_values):
                    # Setting up diagram arrow directions for aquifer to aquifer interactions
                    if net_values[0] > 0:
                        flow_to_1_2 = 0
                        flow_fr_1_2 = 1
                    else:
                        flow_to_1_2 = 1
                        flow_fr_1_2 = 0
                        net_values[0] = abs(net_values[0])
                        
                    if net_values[1] > 0:
                        flow_to_1_3 = 0
                        flow_fr_1_3 = 2
                    else:
                        flow_to_1_3 = 2
                        flow_fr_1_3 = 0
                        net_values[1] = abs(net_values[1])
                         
                    if net_values[2] > 0:
                        flow_to_2_3 = 1
                        flow_fr_2_3 = 2
                    else:
                        flow_to_2_3 = 2
                        flow_fr_2_3 = 1
                        net_values[2] = abs(net_values[2])
                    
                    # Setting up diagram arrow directions for aquifer to stream interactions
                    if net_values[4] > 0:
                        flow_to_1_str = 0
                        flow_fr_str_1 = 4
                    else:
                        flow_to_1_str = 4
                        flow_fr_str_1 = 0
                        net_values[4] = abs(net_values[4])
                        
                    if net_values[6] > 0:
                        flow_to_2_str = 1
                        flow_fr_str_2 = 4
                    else:
                        flow_to_2_str = 4
                        flow_fr_str_2 = 1
                        net_values[6] = abs(net_values[6])
                         
                    if net_values[2] > 0:
                        flow_to_3_str = 2
                        flow_fr_str_3 = 4
                    else:
                        flow_to_3_str = 4
                        flow_fr_str_3 = 2
                        net_values[8] = abs(net_values[8])
                    source = [flow_to_1_2,flow_to_1_3,flow_to_2_3,0,flow_to_1_str,1,flow_to_2_str,2,flow_to_3_str]
                    target = [flow_fr_1_2,flow_fr_1_3,flow_fr_2_3,3,flow_fr_str_1,3,flow_fr_str_2,3,flow_fr_str_3]
                    return source,target
                
                source_1,target_1 = calc_net_flow_directions(net_values_1)
                
                # Grab the emulated ZB data
                disturbed_forecast = forecast_results.copy()
                
                # For layer 1
                disturbed_forecast_1 = disturbed_forecast.loc[
                        (obs.oname == "zbg") & (obs.zone == '1') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                
                # For layer 2
                disturbed_forecast_2 = disturbed_forecast.loc[
                        (obs.oname == "zbg") & (obs.zone == '2') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                
                # For layer 3
                disturbed_forecast_3 = disturbed_forecast.loc[
                        (obs.oname == "zbg") & (obs.zone == '3') & (obs.kper == f'{SP}') & 
                        (~obs.usecol.isin(['from-zone-0', 'to-zone-0'])), :].copy()
                
                # Values for Sankey diagram - for forecasted data
                values = []
                ## Append data from layer 1
                [values.append(x) for x in disturbed_forecast_1.iloc[[8,9,4,5,6]]['forecast'].values]

                # Append data from layer 2
                [values.append(x) for x in disturbed_forecast_2.iloc[[7,9,4,5,6]]['forecast'].values]

                # Append data from layer 3
                [values.append(x) for x in disturbed_forecast_3.iloc[[7,8,4,5,6]]['forecast'].values]

                
                # Now calculate net flows
                net_values_2 = []
                # net flow 1->2, 1->3, 2->3, and then aquifer->tributary and aquifer->stream for each layer
                [net_values_2.append(x) for x in [values[0]-values[5],      # 1-->2
                                                  values[1]-values[10],      # 1-->3
                                                  values[6]-values[11],      # 2-->3
                                                  values[2],values[4]-values[3],      # Layer 1
                                                  values[7],values[9]-values[8],      # Layer 2
                                                  values[12],values[14]-values[13]    # Layer 3
                                                  ]
                 ]
                source_2,target_2 = calc_net_flow_directions(net_values_2)
                link_colors = []
                for i in range(9):
                    link_colors.append('rgba(69, 167, 247,0.7)')
                    link_colors.append('rgba(245, 175, 47,0.6)')
                
                # Make final source, target, and values for Sankey diagram
                source = []
                target = []
                values = []
                for i in range(len(source_1)):
                    source.append(source_1[i])
                    source.append(source_2[i])
                    target.append(target_1[i])
                    target.append(target_2[i])
                    values.append(net_values_1[i])
                    values.append(net_values_2[i])
                
                # Show net flows, for baseline and forecast
                fig = go.Figure(data=[go.Sankey(
                    arrangement = "snap",
                    node = dict(
                      pad = 15,
                      thickness = 20,
                      line = dict(color = "black", width = 0.5),
                      #             0            1           2               3             4
                      label = ["Aquifer 1", "Aquifer 2", "Aquifer 3", "Main Tributary", "River"],
                      #x = [0.1, 0.5, 0.9, 0.7, 0.7],
                      #y = [0.5, 0.5, 0.5, 0.99, 0.1],
                      #color = "blue"
                    ),
                    link = dict(
                        arrowlen=15,
                                 #0,1,2,3,4,5,6,7,8,9,10,11
                        source = source,
                        target = target,
                        value = values,
                        color = link_colors
                  ))])

                # Match the webpage background color
                fig.layout.paper_bgcolor = '#F2F2F2'
                fig.layout.plot_bgcolor = '#DAEDFF'
                fig.update_traces(valueformat=".1e")
                fig.update_layout(title_text="Zone Budget Flows: <span style='color:blue'>Blue</span> arrows indicate baseline scenario and <span style='color:#db8d48'>Orange</span> arrows are emulated <br>All flows are in units of m<sup>3</sup>/day",
                                  font_size=14)
                return fig,{'display':'none'},{'display':'none'},{'display':'block'}
            
            # -----------------------
            # ---- zonbud time-series
            # -----------------------
            elif zonbud_option == 'time_series':
                zb_response = zonbud_dropdown
                fig = make_subplots()
                
                # Calculate the Forecast
                disturbed_forecast = forecast_results.copy()
                
                # Need to sort the index of the results based on obs
                _df = obs.loc[(obs.oname == "zbg") & (obs.usecol == zb_response) & (obs.zone == zonbud_ts_aquifer), :].copy()
                _df.totim = _df.totim.astype(float)
                _df.sort_values("totim",inplace=True)
                disturbed_forecast = disturbed_forecast.loc[(obs.oname == "zbg") & (obs.usecol == zb_response) & (obs.zone == zonbud_ts_aquifer), :].copy()
                disturbed_forecast = disturbed_forecast.reindex(_df.index)
                times = [float(i.split("totim:")[1].split("_")[0]) for i in disturbed_forecast.index]               
                start_date = pd.Timestamp('2022-01-01')
                dates = [start_date + timedelta(days=t) for t in times]
                
                # Plot cumulative sum of values, if specified
                if switch:
                    disturbed_forecast[['modelled','forecast']] = disturbed_forecast[['modelled','forecast']].cumsum()
                
                # Plot the original and disturbed timeseries
                # Only on first round if > 1 multipliers
                fig.add_trace(go.Scatter(x=dates,
                                         y=disturbed_forecast.modelled,
                                         mode='lines',
                                         name='original',
                                         line=dict(color='black',
                                                   )
                                         ),
                              )
                
                # Plot disturbed forecast
                fig.add_trace(go.Scatter(x=dates,
                                         y=disturbed_forecast.forecast,
                                         mode='lines',
                                         name='Scenario Results',
                                         line=dict(dash='dash'),
                                         #line=dict(color='red'),
                                         )
                              )
                layout = go.Layout(title=f'Observation Timeseries Group: {response}',
                                   yaxis=dict(title='Rate (m<sup>3</sup>/day)'),
                                   xaxis=dict(title='Date'))
                fig.update_layout(layout)
                # Match the webpage background color
                fig.layout.paper_bgcolor = '#F2F2F2'
                fig.layout.plot_bgcolor = '#DAEDFF'
                fig.update_layout(yaxis=dict(tickformat=".1e"))
                return fig,{'display':'none'},{'display':'none'},{'display':'block'}
            
        # ---------------------------
        # ---- Simulated Head contour
        # ---------------------------
        elif response == 'sim_hds':
            x = np.load(os.path.join(md,'master','x.npy'))
            y = np.load(os.path.join(md,'master','y.npy'))
            # Model dis
            nrow = 120
            ncol = 60
            # heads in all cells in layer 0 at the end of the simulation
            _df = obs.loc[obs.obgnme==f"hds_layer{layer_slider}_kper{SP_slider}",:].copy()
            _df["i"]=_df["i"].astype(int)
            _df["j"]=_df["j"].astype(int)
            _df = _df.sort_values(by=["i","j"])
            disturbed_forecast = forecast_results.copy()
            # disturbed_forecast = disturbed_forecast.set_index('name')
            disturbed_forecast = disturbed_forecast.loc[obs.obgnme==f"hds_layer{layer_slider}_kper{SP_slider}",:].copy()
            disturbed_forecast = disturbed_forecast.reindex(_df.index)
            disturbed_forecast = disturbed_forecast.replace(1e30, np.nan)
  
            # Make the subplot figure
            fig = make_subplots(rows=1,
                                cols=3,
                                subplot_titles=('Base Case',
                                                'Forecast',
                                                'Forecast-Base'))
            
            # get well locations for plotting
            # For pumping wells:
            wel_x = []
            wel_y = []
            well_data = np.load(os.path.join(md,'master','well_data.npy'),
                                allow_pickle=True)
            for well in well_data:
                wel_x.append(x[0,:][well[0][2]])
                wel_y.append(y[:,0][well[0][1]])
                
            # For monitoring wells:
            trgw = ['trgw-0-29-5', 'trgw-0-41-32', 'trgw-0-68-47', 
                    'trgw-0-8-29','trgw-0-80-20', 'trgw-0-89-47']
            wel_m_x = []
            wel_m_y = []
            for well in trgw:
                wel_m_x.append(x[0,:][int(well.split('-')[3])])
                wel_m_y.append(y[:,0][int(well.split('-')[2])])
                
            # Get the vmin and vmax for plotting
            vmin = min(disturbed_forecast.modelled.min(),
                       disturbed_forecast.forecast.min())

            vmax = max(disturbed_forecast.modelled.max(),
                       disturbed_forecast.forecast.max())

            # Base Case
            a = disturbed_forecast.modelled.values.reshape((nrow,ncol))
            # Plot head contour
            fig.add_trace(go.Contour(z=a,
                                     x=x[0, :],
                                     y=y[:, 0],
                                     colorscale='viridis',
                                     name='Head Contour',
                                     colorbar=dict(
                                                tickvals=[],  # Hide ticks
                                                ticktext=[]   # Hide tick text
                                            ),
                                     zmin=vmin,
                                     zmax=vmax
                                     ),
                          row=1,
                          col=1)
            
            # Plot the SFR stream, in this case it is all of column 47
            s = np.full(a.shape, np.nan)
            s[:,46] = 10
            s[:,47] = 10
            s[:,48] = 10
            custom_color = [[0, 'blue'], [1, 'blue']]
            fig.add_trace(go.Contour(z=s,
                                     x=x[0, :],
                                     y=y[:, 0],
                                     colorscale=custom_color,
                                     name='SFR Stream',
                                     colorbar=dict(
                                                tickvals=[],  # Hide ticks
                                                ticktext=[]   # Hide tick text
                                            ),
                                     ),
                          row=1,
                          col=1)
            
            # Plot pumping well locations
            fig.add_trace(go.Scatter(x=wel_x,
                                     y=wel_y,
                                     mode='markers',
                                     name='Pumping Well',
                                     marker=dict(color='#FFFFFF',
                                                 size=10,
                                                 line=dict(width=2,
                                                           color='black')),
                                     showlegend=False,
                                     ),
                          row=1,
                          col=1
                          )
            # Plot monitroing well locations
            fig.add_trace(go.Scatter(x=wel_m_x,
                                     y=wel_m_y,
                                     mode='markers',
                                     name='Monitoring Well',
                                    marker=dict(color='red',
                                                size=10,
                                                line=dict(width=2,
                                                          color='black')),
                                     showlegend=False
                                     ),
                          row=1,
                          col=1
                          )
            
            # Forecasted Case
            b = disturbed_forecast.forecast.values.reshape((nrow,ncol))
            fig.add_trace(go.Contour(z=b,
                                     x=x[0, :],
                                     y=y[:, 0],
                                     colorscale='viridis',
                                     name='Head Contour',
                                     colorbar={"title": {
                                                         "text": "Modeled Head (m)",
                                                         "side": "right",
                                                         "font": {"size": 16}
                                                         },
                                               'x':-0.15,
                                              },
                                     zmin=vmin,
                                     zmax=vmax
                                     ),
                          row=1,
                          col=2)
            # Plot pumping well locations
            fig.add_trace(go.Scatter(x=wel_x,
                                     y=wel_y,
                                     mode='markers',
                                     name='Pumping Well',
                                     marker=dict(color='#FFFFFF',
                                                 size=10,
                                                 line=dict(width=2,
                                                           color='black')),
                                     showlegend=False
                                     ),
                          row=1,
                          col=2
                          )
            # Plot monitroing well locations
            fig.add_trace(go.Scatter(x=wel_m_x,
                                     y=wel_m_y,
                                     mode='markers',
                                     name='Monitoring Well',
                                     marker=dict(color='red',
                                                 size=10,
                                                 line=dict(width=2,
                                                           color='black')),
                                     showlegend=False
                                     ),
                          row=1,
                          col=2
                          )
            
            # Differnce: forecast - base
            c = b - a
            c = np.where(abs(c)<0.01,0,c)
            fig.add_trace(go.Contour(z=c,
                                     x=x[0, :],
                                     y=y[:, 0],
                                     colorscale='Spectral',
                                     name='Head Contour',
                                     colorbar={"title": {
                                                         "text": "Head Difference (m)",
                                                         "side": "right",
                                                         "font": {"size": 16}
                                                        },
                                               },
                                     ),
                          row=1,
                          col=3)
            # Plot pumping well locations
            fig.add_trace(go.Scatter(x=wel_x,
                                     y=wel_y,
                                     mode='markers',
                                     name='Pumping Well',
                                     marker=dict(color='#FFFFFF',
                                                 size=10,
                                                 line=dict(width=2,
                                                           color='black')),
                                     showlegend=True
                                     ),
                          row=1,
                          col=3,
                          )
            # Plot monitroing well locations
            fig.add_trace(go.Scatter(x=wel_m_x,
                                     y=wel_m_y,
                                     mode='markers',
                                     name='Monitoring Well',
                                     marker=dict(color='red',
                                                 size=10,
                                                 line=dict(width=2,
                                                           color='black')),
                                     showlegend=True
                                     ),
                          row=1,
                          col=3
                          )
            fig.update_layout(
                xaxis=dict(range=[x.min(), x.max()], scaleanchor='y', constrain='domain'),
                yaxis=dict(range=[0, 10000], scaleratio=1),
                legend_orientation="h",
                )
            # To ensure all traces fit within the same aspect ratio settings
            fig.update_xaxes(scaleanchor='y', constrain='domain')
            fig.update_yaxes(scaleanchor='x', scaleratio=1)
            
            return fig,{'display':'none'},{'display':'block'},{'display':'none'}
        
    
        # --------------------------
        # ---- Monitoring well heads
        # --------------------------
        elif response == 'monitoring_well_hds':
            trgw = ['trgw-0-29-5', 'trgw-0-41-32', 'trgw-0-68-47', 
                    'trgw-0-8-29','trgw-0-80-20', 'trgw-0-89-47']
            idx = [[1,1],[1,2],
                   [2,1],[2,2],
                   [3,1],[3,2]]
            idx_ref = dict(zip(trgw,idx))
            subplot_titles = ['Layer '+ str(int(x.split('-')[1])+1) + ', Row ' + str(int(x.split('-')[2])+1) + ', Column ' + str(int(x.split('-')[3])+1) for x in trgw]
            fig = make_subplots(3,2,
                                subplot_titles=subplot_titles,
                                shared_yaxes=True,
                                y_title='Modeled Head (m)',
                                x_title='Date')
            # One plot for each well
            for j,well in enumerate(trgw):
                if j == 0:
                    leg = True
                else:
                    leg = False
                r = idx_ref[well][0]
                c = idx_ref[well][1]
                
                # Consistent colors, assuming no more than 10 multipliers
                colors = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A',
                          '#19D3F3','#FF6692','#B6E880','#FF97FF','#FECB52']
                
                disturbed_forecast = forecast_results.copy()
                # disturbed_forecast = disturbed_forecast.set_index('name')
                disturbed_forecast = disturbed_forecast.loc[obs.usecol==well].copy()
                times = [float(i.split("time:")[-1]) for i in disturbed_forecast.index]
                start_date = pd.Timestamp('2022-01-01')
                dates = [start_date + timedelta(days=t) for t in times]
                
                # Plot cumulative sum of values, if specified
                if switch:
                    disturbed_forecast[['modelled','forecast']] = disturbed_forecast[['modelled','forecast']].cumsum()
                
                # Plot the original and disturbed timeseries
                # Only on first round if > 1 multipliers
                fig.add_trace(go.Scatter(x=dates,
                                         y=disturbed_forecast.modelled,
                                         mode='lines',
                                         name='original',
                                         line=dict(color='black',
                                                   ),
                                         showlegend=leg
                                         ),
                              row=r,
                              col=c
                              )
                
                # Plot disturbed forecast
                fig.add_trace(go.Scatter(x=dates,
                                         y=disturbed_forecast.forecast,
                                         mode='lines',
                                         name='Scenario Results',
                                         showlegend=leg,
                                         line=dict(color=colors[0], dash='dash')
                                         ),
                              row=r,
                              col=c
                              )
                    
            # Match the webpage background color
            fig.layout.paper_bgcolor = '#F2F2F2'
            fig.layout.plot_bgcolor = '#DAEDFF'
            return fig,{'display':'block'},{'display':'none'},{'display':'none'}


# -----------------------------------------------------
# Callback to display Modal of selected well timeseries
# -----------------------------------------------------
@app.callback(Output('modal','is_open'),
              Output('modal_graph','figure'),
              Output('modal_body','children'),
              Input('graph-output','clickData'),
              State('modal','is_open'),
              State('response_selector','value'),
              prevent_initial_call=True)
                  
def toggle_modal(click,is_open,response_var):
    global forecast_results
    if response_var=='sim_hds':
        # Determine coordinates of the click
        click_coords = [click['points'][0]['x'],click['points'][0]['y']]
        
        # Load GW model and determine well locations
        x = np.load(os.path.join(md,'master','x.npy'))
        y = np.load(os.path.join(md,'master','y.npy'))
        well_plot = False
        
        # I believe these are monitoring wells...
        trgw = ['trgw-0-29-5', 'trgw-0-41-32', 'trgw-0-68-47', 
                'trgw-0-8-29','trgw-0-80-20', 'trgw-0-89-47']
        
        # Determine which (or if) well was clicked
        for well in trgw:
            _x = x[0,:][int(well.split('-')[3])]
            _y = y[:,0][int(well.split('-')[2])]
            if click_coords == [_x,_y]:
                well_plot = well
                break
        
        # If a well is clicked, make the corresponding well figure
        if well_plot:
            # Heads timeseries
            fig = make_subplots()
            
            # Load the forecasted DataFrame
            disturbed_forecast = forecast_results.copy() 
            disturbed_forecast = disturbed_forecast.loc[obs.usecol==well_plot].copy()
            times = [float(i.split("time:")[-1]) for i in disturbed_forecast.index]
            start_date = pd.Timestamp('2022-01-01')
            dates = [start_date + timedelta(days=t) for t in times]
            
            # Plot the original and disturbed timeseries
            # Only on first round if > 1 multipliers
            fig.add_trace(go.Scatter(x=dates,
                                     y=disturbed_forecast.modelled,
                                     mode='lines',
                                     name='original',
                                     line=dict(color='black'),
                                     ),
                          )
            
            # Plot disturbed forecast
            fig.add_trace(go.Scatter(x=dates,
                                     y=disturbed_forecast.forecast,
                                     mode='lines',
                                     line=dict(dash='dash'),
                                     #line=dict(color='red'),
                                     )
                          )
            # Layout
            layout = go.Layout(#title=f'Observation Timeseries Group: {response}',
                               yaxis=dict(title='Well Head (m)'),
                               xaxis=dict(title='Date'))
            fig.update_layout(layout)
            # Match the webpage background color
            fig.layout.paper_bgcolor = '#F2F2F2'
            fig.layout.plot_bgcolor = '#DAEDFF'
            
            if click:
                return not is_open,fig,f'Well: {well_plot}'
            return is_open,fig,f'Well: {well_plot}'
        else:
            return False,{},'Temp'
    else:
        return False,{},'Temp'
    
    
# -------------------
# Start the dashboard
# -------------------
if __name__ == "__main__":
    app.run(debug=False,port=8050)











