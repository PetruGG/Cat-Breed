import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from translate import Translator
from sklearn.preprocessing import LabelEncoder
import random


def convertToNumeric(df):
    text_to_numeric_mappings = {}
    non_numeric_cols = df.select_dtypes(include='object').columns

    for col in non_numeric_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        text_to_numeric_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    df.to_csv('./datasets/Numeric_Dataset.csv', index=False)
    print("Numeric conversion complete")
    # print(df.head())
    # print("Text to Numeric Mappings:", text_to_numeric_mappings)
    return text_to_numeric_mappings


def showCorrelations(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(30, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlations between Attributes and Classes")
    plt.show()


def translate(dataset, source_lang, target_lang):
    df = dataset
    translator = Translator(to_lang=target_lang, from_lang=source_lang)
    column_to_translate = "Additional Comments"
    df[column_to_translate] = df[column_to_translate].apply(
        lambda text: translator.translate(text) if pd.notna(text) and text != "" else text
    )
    df.to_csv("./datasets/Translated_Dataset.csv", index=False)
    print("Translation complete!")


def initial_analysis(dataset):
    df = dataset
    print("Missing data verification")
    print(df.isnull().sum())
    print("Duplicate data verification")
    print(df.duplicated().sum())
    print("Number of instances per class")
    print(df['Breed'].value_counts())


def extract_distinct_value_list(dataset):
    distinct_values_count = {}
    for column in dataset.columns:
        distinct_values = dataset[column].dropna().unique()
        distinct_values_count[column] = {
            'Total Distinct Values': len(distinct_values),
            'Distinct Values': distinct_values.tolist()
        }
    print("Total Distinct Values at File Level:")
    for column, values in distinct_values_count.items():
        print(f"{column}: {values['Total Distinct Values']}")
    print("\nDistinct Values per Class (by Breed):")
    distinct_values_per_class = dataset.groupby('Breed').nunique()
    for breed, row in distinct_values_per_class.iterrows():
        print(f"{breed}:")
        for attribute, count in row.items():
            print(f" {attribute}: {count}")


def histogram(dataset):
    for i, column in enumerate(dataset.columns[2:-1], start=1):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        sns.histplot(dataset[column], bins=5, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def countplot(dataset):
    for i, column in enumerate(dataset.columns[2:-1], start=1):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        sns.countplot(x=dataset[column])
        plt.title(f'Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def generate_samples_by_breed(df, breed, num_samples):
    breed_df = df[df['Breed'] == breed]
    generated_samples = []
    existing_rows = set([tuple(row) for row in breed_df.values])

    if len(breed_df) >= num_samples:
        return breed_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        sampled_df = breed_df.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        for _, row in sampled_df.iterrows():
            new_row = row.copy()

            for col in numerical_columns:
                unique_found = False
                attempt = 0
                while not unique_found and attempt < 10:
                    new_value = round(row[col] + random.uniform(-0.5, 0.5))
                    new_row[col] = new_value
                    attempt += 1
                    if tuple(new_row) not in existing_rows:
                        unique_found = True

            for col in categorical_columns:
                unique_found = False
                attempt = 0
                unique_values = breed_df[col].unique()
                while not unique_found and attempt < 10:
                    new_value = random.choice(unique_values)
                    new_row[col] = new_value
                    attempt += 1
                    if tuple(new_row) not in existing_rows:
                        unique_found = True

            existing_rows.add(tuple(new_row))
            generated_samples.append(new_row)

        return pd.DataFrame(generated_samples)


def balance_classes(df, n_samples):
    balanced_df = pd.DataFrame()
    for breed in df['Breed'].unique():
        breed_samples = generate_samples_by_breed(df, breed, n_samples)
        balanced_df = pd.concat([balanced_df, breed_samples], ignore_index=True)
    return balanced_df
