#!/usr/bin/env python
# coding: utf-8

# In[1]:


import base64
import io
import pathlib

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import tifffile
from dash import Dash, Input, Output, State, callback_context, dcc, html
from PIL import Image, ImageEnhance

# In[2]:


def encode_composite_image(
    img_paths,
    contrasts=[1.0, 1.0, 1.0, 1.0],
    colors=[(0, 255, 0), (0, 255, 128), (255, 0, 0), (0, 0, 255)],
):
    """Create composite image with different color channels and contrasts."""
    channels = []

    # Map the channel names to the expected order
    channel_order = ["CL_488_1", "CL_488_2", "CL_561", "DNA"]

    for i, channel_name in enumerate(channel_order):
        if channel_name not in img_paths:
            print(f"Warning: {channel_name} not found in img_paths")
            continue

        img_path = img_paths[channel_name]

        # Load image
        img = tifffile.imread(img_path)
        print(
            f"Loaded {channel_name}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}"
        )

        # Normalize to 0-255 range
        if img.max() > img.min():
            img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                np.uint8
            )
        else:
            img_norm = np.zeros_like(img, dtype=np.uint8)

        print(f"After normalization: min={img_norm.min()}, max={img_norm.max()}")

        # Apply contrast
        pil_img = Image.fromarray(img_norm)
        if contrasts[i] != 1.0:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrasts[i])

        # Convert back to array and apply color
        img_array = np.array(pil_img)
        colored_channel = np.zeros(
            (img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8
        )

        # Apply color channel - FIXED: proper color application
        color = colors[i]
        for c in range(3):
            if color[c] > 0:  # Only apply to non-zero color channels
                colored_channel[:, :, c] = (
                    (img_array.astype(np.float32) / 255.0) * color[c]
                ).astype(np.uint8)

        channels.append(colored_channel)
        print(
            f"Channel {channel_name} processed: color={color}, max_intensity={colored_channel.max()}"
        )

    if not channels:
        # Return a placeholder image if no channels found
        placeholder = np.full((150, 150, 3), 128, dtype=np.uint8)
        pil_composite = Image.fromarray(placeholder)
    else:
        # Composite the channels using additive blending
        composite = np.zeros_like(channels[0], dtype=np.float32)
        for channel in channels:
            composite += channel.astype(np.float32)

        # Clip to valid range and convert back to uint8
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        pil_composite = Image.fromarray(composite)

        print(f"Final composite: min={composite.min()}, max={composite.max()}")

    buffer = io.BytesIO()
    pil_composite.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


# In[3]:


umap_file_path = pathlib.Path(
    "../../3.generate_umap_and_PCA/results/UMAP/single-cell_profiles_CP_scDINO_umap.parquet"
).resolve(strict=True)
umap_df = pd.read_parquet(umap_file_path)
# make the time column numeric
umap_df["Metadata_Time"] = pd.to_numeric(umap_df["Metadata_Time"])
umap_df["Metadata_Time"] = umap_df["Metadata_Time"].astype(int)
umap_df["Metadata_Time"] = umap_df["Metadata_Time"] * 30


# In[1]:


# Make scatter plot with time slider
time_col = "Metadata_Time"

# Get unique time points
time_points = sorted(umap_df[time_col].unique())

# Create initial figure with first time point
initial_df = umap_df[umap_df[time_col] == time_points[0]]
fig = px.scatter(
    initial_df,
    x="UMAP_0",
    y="UMAP_1",
    hover_data=["Metadata_Well", time_col, "Metadata_track_id"],
    title=f"UMAP at {time_points[0]} minutes",
    color_discrete_sequence=["blue"],
    opacity=0.3,
    size_max=1,
)

# Update layout for better appearance
fig.update_layout(width=800, height=600, title_font_size=16)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="umap-plot",
                    figure=fig,
                    style={"width": "100%", "height": "600px"},
                ),
            ],
            style={"width": "50%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.H4("Time Controls"),
                html.Div(
                    [
                        html.Button(
                            "Play",
                            id="play-button",
                            n_clicks=0,
                            style={
                                "margin": "10px",
                                "padding": "10px 20px",
                                "font-size": "16px",
                            },
                        ),
                        html.Button(
                            "Reset",
                            id="reset-button",
                            n_clicks=0,
                            style={
                                "margin": "10px",
                                "padding": "10px 20px",
                                "font-size": "16px",
                            },
                        ),
                        html.P(
                            id="current-time-display",
                            style={"font-size": "18px", "font-weight": "bold"},
                        ),
                        dcc.Interval(
                            id="interval-component",
                            interval=1000,  # Update every 1 second
                            n_intervals=0,
                            disabled=True,  # Start disabled
                        ),
                        # Hidden div to store state
                        html.Div(
                            id="time-index", children="0", style={"display": "none"}
                        ),
                        html.Div(
                            id="is-playing", children="false", style={"display": "none"}
                        ),
                    ],
                    style={"margin": "20px"},
                ),
                # Optional: Keep manual slider for precise control
                html.Div(
                    [
                        html.Label("Manual Time Control:"),
                        dcc.Slider(
                            id="time-slider",
                            min=0,
                            max=len(time_points) - 1,
                            step=1,
                            value=0,
                            marks={
                                i: str(time_points[i])
                                for i in range(
                                    0, len(time_points), max(1, len(time_points) // 10)
                                )
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"margin": "20px"},
                ),
                html.H4("Channel Controls"),
                html.Div(
                    [
                        html.Label("Green 1 (CL_488_1) Contrast:"),
                        dcc.Slider(
                            id="green1-contrast",
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                        ),
                    ],
                    style={"margin": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Green 2 (CL_488_2) Contrast:"),
                        dcc.Slider(
                            id="green2-contrast",
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                        ),
                    ],
                    style={"margin": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Red (CL_561) Contrast:"),
                        dcc.Slider(
                            id="red-contrast",
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                        ),
                    ],
                    style={"margin": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Blue (DNA) Contrast:"),
                        dcc.Slider(
                            id="blue-contrast",
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            marks={0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                        ),
                    ],
                    style={"margin": "10px"},
                ),
                html.Div(id="image-panel", style={"margin-top": "20px"}),
            ],
            style={"width": "45%", "display": "inline-block", "padding": "20px"},
        ),
    ]
)


# FIXED: Only ONE callback to handle play/pause button
@app.callback(
    [
        Output("interval-component", "disabled"),
        Output("play-button", "children"),
        Output("is-playing", "children"),
    ],
    [Input("play-button", "n_clicks"), Input("reset-button", "n_clicks")],
    [State("is-playing", "children")],
)
def toggle_play_pause(play_clicks, reset_clicks, is_playing_str):
    ctx = callback_context

    if not ctx.triggered:
        return True, "Play", "false"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "reset-button":
        return True, "Play", "false"

    elif button_id == "play-button":
        is_playing = is_playing_str == "true"
        if is_playing:
            return True, "Play", "false"  # Pause
        else:
            return False, "Pause", "true"  # Play

    return True, "Play", "false"  # FIXED: Added missing return


# Callback to update time index based on interval or reset
@app.callback(
    [Output("time-index", "children"), Output("time-slider", "value")],
    [
        Input("interval-component", "n_intervals"),
        Input("reset-button", "n_clicks"),
        Input("time-slider", "value"),
    ],
    [State("time-index", "children"), State("is-playing", "children")],
)
def update_time_index(
    n_intervals, reset_clicks, slider_value, current_index_str, is_playing_str
):
    ctx = callback_context

    if not ctx.triggered:
        return "0", 0

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "reset-button":
        return "0", 0

    elif button_id == "time-slider":
        return str(slider_value), slider_value

    elif button_id == "interval-component" and is_playing_str == "true":
        current_index = int(current_index_str)
        next_index = (current_index + 1) % len(time_points)
        return str(next_index), next_index

    return current_index_str, int(current_index_str)


# Callback to update UMAP plot based on time index
@app.callback(
    [Output("umap-plot", "figure"), Output("current-time-display", "children")],
    [Input("time-index", "children")],
)
def update_umap(time_index_str):
    time_index = int(time_index_str)
    current_time = time_points[time_index]

    print(f"Time changed to: {current_time}")

    # Filter data for selected time point
    filtered_df = umap_df[umap_df[time_col] == current_time]
    print(f"Filtered data has {len(filtered_df)} cells")

    fig = px.scatter(
        filtered_df,
        x="UMAP_0",
        y="UMAP_1",
        hover_data=["Metadata_Well", time_col, "Metadata_track_id"],
        title=f"UMAP at {current_time} minutes. ({len(filtered_df)} cells)",
        color_discrete_sequence=["blue"],
        opacity=0.3,
        size_max=1,
    )

    fig.update_layout(width=800, height=600, title_font_size=16)

    time_display = f"Current Time: {current_time} ({time_index + 1}/{len(time_points)})"

    return fig, time_display


# FIXED: Image display callback
@app.callback(
    Output("image-panel", "children"),
    [
        Input("umap-plot", "clickData"),
        Input("green1-contrast", "value"),
        Input("green2-contrast", "value"),
        Input("red-contrast", "value"),
        Input("blue-contrast", "value"),
    ],
    prevent_initial_call=False,
)
def display_composite_image(
    clickData, green1_contrast, green2_contrast, red_contrast, blue_contrast
):
    if clickData is None:
        return "Click a point to see the composite image."

    # Get the track_id from hover data
    track_id = clickData["points"][0]["customdata"][
        2
    ]  # Metadata_track_id is 3rd in hover_data
    current_time = clickData["points"][0]["customdata"][
        1
    ]  # time_col is 2nd in hover_data

    # Find the cell data using track_id and time
    cell_data = umap_df[
        (umap_df["Metadata_track_id"] == track_id) & (umap_df[time_col] == current_time)
    ]

    if len(cell_data) == 0:
        return html.Div([html.H4("Error: Cell data not found")])

    cell_data = cell_data.iloc[0]
    cell_id = cell_data["Metadata_track_id"]

    print(f"Clicked cell_id: {cell_id} at time: {current_time}")

    # Get image paths from the dataframe
    try:
        DNA_path = pathlib.Path(cell_data["Metadata_Image_FileName_DNA_crop"])
        CL_488_1_path = pathlib.Path(cell_data["Metadata_Image_FileName_CL_488_1_crop"])
        CL_488_2_path = pathlib.Path(cell_data["Metadata_Image_FileName_CL_488_2_crop"])
        CL_561_path = pathlib.Path(cell_data["Metadata_Image_FileName_CL_561_crop"])

        # Only include existing files
        img_paths = {}
        if CL_488_1_path.exists():
            img_paths["CL_488_1"] = CL_488_1_path
        if CL_488_2_path.exists():
            img_paths["CL_488_2"] = CL_488_2_path
        if CL_561_path.exists():
            img_paths["CL_561"] = CL_561_path
        if DNA_path.exists():
            img_paths["DNA"] = DNA_path

        if not img_paths:
            return html.Div(
                [
                    html.H4(f"Error: No image files found for cell {cell_id}"),
                    html.P(f"Time point: {current_time}"),
                ]
            )

        # Set contrasts and colors
        contrasts = [green1_contrast, green2_contrast, red_contrast, blue_contrast]
        colors = [(0, 255, 0), (0, 255, 128), (255, 0, 0), (0, 0, 255)]

        # Generate composite image
        composite_b64 = encode_composite_image(img_paths, contrasts, colors)

        return html.Div(
            [
                html.H4(f"Cell: {cell_id}"),
                html.P(f"Time Point: {current_time}"),
                html.Img(src=composite_b64, style={"max-width": "100%"}),
            ]
        )

    except Exception as e:
        return html.Div(
            [
                html.H4(f"Error loading cell: {cell_id}"),
                html.P(f"Error: {str(e)}"),
            ]
        )


if __name__ == "__main__":
    app.run(jupyter_mode="external")
    # app.run(mode='inline')


# In[ ]:
