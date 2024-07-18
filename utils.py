import pandas as pd
import numpy as np


def int_to_binary(df,col):
    n = int(np.log2(df[col].max()) + 1)

    # Function to convert integers in a column to binary format with fixed length
    def int_to_binary_fixed_length(column):
        # print(column)
        dd = pd.DataFrame({k: list(map(int, f'{i:0{n}b}')) for k,i in column.items()}).T
        dd.index.name=  'flow'
        return dd

    # Apply the function to each specified column and concatenate the results
    # columns_to_convert = [col]
    # binary_df_list = 
    binary_df = int_to_binary_fixed_length(df[col]).add_prefix(f'{col}_')
    # print(df)
    # df=df.drop(col)
    # display(binary_df.head())
    # display(df.head())
    df=df.drop(col,axis=1)
    return pd.merge(df,binary_df, left_index=True, right_index=True)
def payload_to_binary_columns(df, column_name,MAX_PAYLOAD_LENGTH):
    """
    Converts a column with binary data into separate bit columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The transformed DataFrame with bit columns.
    """
    # Find the maximum length of the payload
    # max_payload_length = df[column_name].apply(len).max()
    bit_columns = ['bit_' + str(i) for i in range(MAX_PAYLOAD_LENGTH * 8)]
    
    def bytes_to_bits(byte_data):
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        return np.unpackbits(byte_array)
    
    # Convert and pad the binary data
    binary_data = df[column_name].apply(lambda x: np.pad(bytes_to_bits(x), (0, MAX_PAYLOAD_LENGTH * 8 - len(x) * 8), 'constant'))

    # Stack the binary data into a matrix
    binary_matrix = np.vstack(binary_data.values)
    
    # Create a DataFrame from the binary matrix
    binary_df = pd.DataFrame(binary_matrix, columns=bit_columns)
    df=df.reset_index()
    # Concatenate the binary DataFrame with the original DataFrame
    return pd.concat([df.drop(columns=[column_name]), binary_df], axis=1).set_index('flow')

def encode_time(df):
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Encode minute cyclically
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Encode second cyclically
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)
    df.drop('time',axis=1,inplace=True)