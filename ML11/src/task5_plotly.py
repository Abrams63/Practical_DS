import numpy as np
import pandas as pd
import plotly.express as px


def create_interactive_plotly(df):
    """
    Creates an interactive scatter plot using Plotly.

    Parameters:
    df (DataFrame): A DataFrame containing 'x' and 'y' columns.

    Returns:
    A Plotly Figure object.
    """
    # Create a scatter plot with title and axis labels
    fig = px.scatter(
        df,
        x='x',
        y='y',
        title='Interactive Scatter Plot',
        labels={'x': 'X Axis', 'y': 'Y Axis'},
        template='plotly_white'
    )
    # Enhance interactivity: configure markers and display the legend
    fig.update_traces(marker=dict(size=10), mode='markers', name='Data Points')
    fig.update_layout(showlegend=True)
    return fig


# Example data
df = pd.DataFrame({'x': np.random.rand(50), 'y': np.random.rand(50)})
create_interactive_plotly(df)
