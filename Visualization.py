import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path_real = "adult.csv"  # Real dataset
file_path_synthetic = "synthetic_data1.csv" 

df_real = pd.read_csv(file_path_real)
df_syn = pd.read_csv(file_path_synthetic)

def clean_income(value):
    value = str(value).strip()  
    if value in ['<50K', '<=50K', '50K']:  
        return '<=50K'
    elif value in ['>50K', '>=50K']: 
        return '>50K'
    elif '<' in value:  
        return '<=50K'
    elif '>' in value:  
        return '>50K'
    elif value == '' or value.lower() == 'nan':  
        return 'Missing'
    else:
        return value  

df_real['income'] = df_real['income'].apply(clean_income)
df_syn['income'] = df_syn['income'].apply(clean_income)

subset_df = df_real.sample(n=150, random_state=42)
subset_df.to_csv("subset_adult.csv", index=False)

df_real = df_real.drop(columns=['workclass', 'fnlwgt'], errors='ignore')

def plot_age_distribution(df, title):
    plt.figure(figsize=(10, 6))
    for income_group in df['income'].unique():
        subset = df[df['income'] == income_group]
        plt.hist(subset['age'], bins=20, alpha=0.5, label=f"Income: {income_group}")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(title="Income")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_age_distribution(df_real, "Age Distribution by Income Group (Real Data)")
plot_age_distribution(df_syn, "Age Distribution by Income Group (Synthetic Data)")

age_bins = [18, 25, 35, 45, 55, 65, 75, 100]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '75+']

df_real['age_group'] = pd.cut(df_real['age'], bins=age_bins, labels=age_labels, right=False)
df_syn['age_group'] = pd.cut(df_syn['age'], bins=age_bins, labels=age_labels, right=False)

def plot_age_group_distribution(df, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='age_group', hue='income', palette='viridis')
    plt.title(title)
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.legend(title="Income")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_age_group_distribution(df_real, "Income Distribution Across Age Groups (Real Data)")
plot_age_group_distribution(df_syn, "Income Distribution Across Age Groups (Synthetic Data)")

def plot_education_distribution(df, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='education', hue='income', palette='Set2', order=df['education'].value_counts().index)
    plt.title(title)
    plt.xlabel("Education")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Income")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_education_distribution(df_real, "Income Distribution Across Education Levels (Real Data)")
plot_education_distribution(df_syn, "Income Distribution Across Education Levels (Synthetic Data)")


def plot_race_distribution(df, title):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='race', hue='income', palette='husl')
    plt.title(title)
    plt.xlabel("Race")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Income")
    plt.show()

plot_race_distribution(df_real, "Income Distribution Across Race Groups (Real Data)")
plot_race_distribution(df_syn, "Income Distribution Across Race Groups (Synthetic Data)")

def plot_gender_distribution(df, title):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sex', hue='income', palette='coolwarm')
    plt.title(title)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.legend(title="Income")
    plt.show()

plot_gender_distribution(df_real, "Income Distribution by Gender (Real Data)")
plot_gender_distribution(df_syn, "Income Distribution by Gender (Synthetic Data)")

def plot_boxplots(df, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='income', y='age', data=df, palette='Set2')
    plt.title(f"Boxplot of Age by Income ({title})")
    plt.xlabel("Income")
    plt.ylabel("Age")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='income', y='hours-per-week', data=df, palette='Set2')
    plt.title(f"Boxplot of Hours Per Week by Income ({title})")
    plt.xlabel("Income")
    plt.ylabel("Hours Per Week")
    plt.show()

plot_boxplots(df_real, "Real Data")
plot_boxplots(df_syn, "Synthetic Data")
