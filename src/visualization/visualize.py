import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить данные из файла."""
    return pd.read_csv(file_path)

def plot_survival_by_feature(df: pd.DataFrame, feature: str) -> None:
    """Построить график выживаемости по заданному признаку."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature, y='Survived', data=df, ci=None)
    plt.title(f'Survival by {feature}')
    plt.show()

def main():
    processed_data_path = os.path.join('data', 'processed', 'precessed_train.csv')
    df = load_data(processed_data_path)
    plot_survival_by_feature(df, 'Pclass')
    plot_survival_by_feature(df, 'Sex')
    plot_survival_by_feature(df, 'Embarked')
    plot_survival_by_feature(df, 'Age')

if __name__ == '__main__':
    main()

