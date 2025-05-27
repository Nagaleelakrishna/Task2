import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("N:\AIML (Elavate Labs )\Titanic-Dataset.csv")
print(df.describe())
print(df.isnull().sum())
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
sns.pairplot(df[numeric_cols])
plt.show()
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True)
plt.show()
for col in numeric_cols:
    fig = px.box(df, y=col, title=col)
    fig.show()
for col in numeric_cols:
    print(col, "Mean:", df[col].mean(), "Median:", df[col].median(), "Std:", df[col].std())