import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить данные из файла."""
    return pd.read_csv(file_path)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Сохранить данные в файл."""
    df.to_csv(file_path, index=False)

def main():
    raw_train_data_path = os.path.join('data', 'raw', 'raw_train.csv')
    processed_train_data_path = os.path.join('data', 'processed', 'processed_train.csv')
  
    raw_test_data_path = os.path.join('data', 'raw', 'raw_test.csv')
    processed_test_data_path = os.path.join('data', 'processed', 'processed_test.csv')
    
    df_train = load_data(raw_data_path)
    # Обработка данных
    # Заполнение пропусков, кодирование категориальных признаков и т.д.
    df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
    df_train = pd.get_dummies(df_train, columns=['Sex', 'Embarked'], drop_first=True)
    
    features_list = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

    df_train = df_train[features_list]
    
    save_data(df_train, processed_train_data_path)

    df_test = load_data(raw_data_path)
    # Обработка данных
    # Заполнение пропусков, кодирование категориальных признаков и т.д.
    df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
    df_test['Embarked'].fillna(df_test['Embarked'].mode()[0], inplace=True)
    df_test = pd.get_dummies(df_test, columns=['Sex', 'Embarked'], drop_first=True)

    df_test = df_test[features_list]
    
    save_data(df_test, processed_test_data_path)

if __name__ == '__main__':
    main()

