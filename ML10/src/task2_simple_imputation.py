import pandas as pd


def impute_missing_values(df):
    """
    Impute missing values in the dataset using mean, median, and mode.

    Parameters:
    df (pd.DataFrame): DataFrame with missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    
    return df


# Example data
data = {
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': ['cat', None, 'dog', 'dog']
}
df = pd.DataFrame(data)

# Apply the function to the dataset
df_imputed = impute_missing_values(df)
print(df_imputed)
