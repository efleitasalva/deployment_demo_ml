import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from typing import Union

def process_data(
    df: pd.DataFrame, 
    numeric_features: list[str], 
    categorical_features: list[str], 
    target_column: str = None
) -> tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Procesa los datos transformando las columnas numéricas y categóricas.
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        numeric_features (list[str]): Lista de nombres de columnas numéricas.
        categorical_features (list[str]): Lista de nombres de columnas categóricas.
        target_column (str, opcional): Nombre de la columna objetivo. Por defecto es None.

    Returns:
        tuple[pd.DataFrame, Union[pd.Series, None]]: 
            Un DataFrame con las transformaciones aplicadas y la columna objetivo (si se especifica).
    """
    # Extraer columna objetivo si se especifica
    target = df[target_column] if target_column else None

    # Filtrar la columna objetivo de las listas de características
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Crear una copia del DataFrame sin la columna objetivo
    df_features = df.drop(columns=[target_column], axis=1) if target_column else df.copy()

    # Escalar columnas numéricas
    scaler = RobustScaler()
    df_numeric = pd.DataFrame(
        scaler.fit_transform(df_features[numeric_features]), 
        columns=numeric_features, 
        index=df.index
    )

    # Codificar columnas categóricas
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_categorical = pd.DataFrame(
        encoder.fit_transform(df_features[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=df.index
    )

    # Combinar columnas transformadas
    df_processed = pd.concat([df_numeric, df_categorical], axis=1)

    # Imputar valores nulos en la columna 'Financial Stress' si existe
    if 'Financial Stress' in df_processed.columns:
        df_processed['Financial Stress'] = df_processed['Financial Stress'].fillna(
            df_processed['Financial Stress'].mode()[0]
        )
    
    print("Data processing finished")
    return df_processed, target
