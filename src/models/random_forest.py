import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить данные из файла."""
    return pd.read_csv(file_path)

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Обучить модель Random Forest."""
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    
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

