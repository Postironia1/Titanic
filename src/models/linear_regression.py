import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить данные из файла."""
    return pd.read_csv(file_path)

def train_model(df: pd.DataFrame) -> LinearRegression:
    """Обучить модель линейной регрессии."""
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    
    print(f'Accuracy on train: {accuracy:.4f}')
    
    return model

def make_submission(df: pd.DataFrame, model: LinearRegression) -> pd.DataFrame:
    """Подготовить датасет для предикта"""
    X = df.drop(columns=[Survived])
    y = model.predict(X)
    df['Survived'] = (y_pred >= 0.5).astype(int)  
    return df[['PassengerId', 'Survived']]

def main():
    processed_train_data_path = os.path.join('data', 'processed', 'precessed_train.csv')
    processed_test_data_path = os.path.join('data', 'processed', 'precessed_test.csv')
  
    df_train = load_data(processed_data_path)
    df_test = load_data(processed_data_path)
  
    train_model(df_train)
  
    # Предсказание на тестовых данных
    df_test_predicted = predict(df_test, model)

if __name__ == '__main__':
    main()

