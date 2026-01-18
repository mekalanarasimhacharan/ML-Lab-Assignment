import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def load_purchase_data(file_path):
    purchase_df = pd.read_excel(file_path, sheet_name="Purchase data")
    input_matrix = purchase_df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    payment_vector = purchase_df["Payment (Rs)"].values.reshape(-1, 1)
    return input_matrix, payment_vector

def get_vector_space_dimension(input_matrix):
    return input_matrix.shape[1]

def get_number_of_vectors(input_matrix):
    return input_matrix.shape[0]

def calculate_matrix_rank(input_matrix):
    return np.linalg.matrix_rank(input_matrix)

def calculate_cost_using_pseudo_inverse(input_matrix, payment_vector):
    pseudo_inverse_matrix = np.linalg.pinv(input_matrix)
    item_cost = pseudo_inverse_matrix.dot(payment_vector)
    return item_cost

def classify_customers(payment_vector, threshold=200):
    classification = []
    for amount in payment_vector:
        if amount > threshold:
            classification.append("RICH")
        else:
            classification.append("POOR")
    return classification

def load_stock_data(file_path):
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

def mean_variance_numpy(price_values):
    return np.mean(price_values), np.var(price_values)

def manual_mean(price_values):
    total_sum = 0
    for value in price_values:
        total_sum += value
    return total_sum / len(price_values)

def manual_variance(price_values):
    avg_value = manual_mean(price_values)
    variance_sum = 0
    for value in price_values:
        variance_sum += (value - avg_value) ** 2
    return variance_sum / len(price_values)

def average_time_taken(function_name, values):
    execution_times = []
    for _ in range(10):
        start_time = time.time()
        function_name(values)
        end_time = time.time()
        execution_times.append(end_time - start_time)
    return sum(execution_times) / len(execution_times)

def probability_of_loss(change_percentages):
    negative_changes = list(filter(lambda x: x < 0, change_percentages))
    return len(negative_changes) / len(change_percentages)

def probability_profit_on_wednesday(stock_df):
    wednesday_records = stock_df[stock_df["Day"] == "Wednesday"]
    if len(wednesday_records) == 0:
        return 0
    profit_days = wednesday_records[wednesday_records["Chg%"] > 0]
    return len(profit_days) / len(wednesday_records)

def conditional_probability_profit_wednesday(stock_df):
    wednesday_records = stock_df[stock_df["Day"] == "Wednesday"]
    if len(wednesday_records) == 0:
        return 0
    profit_days = wednesday_records[wednesday_records["Chg%"] > 0]
    return len(profit_days) / len(wednesday_records)

def plot_change_vs_day(stock_df):
    plt.scatter(stock_df["Day"], stock_df["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Change Percentage")
    plt.title("Change Percentage vs Day")
    plt.show()

def load_thyroid_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

def analyze_thyroid_data(thyroid_df):
    data_summary = thyroid_df.describe(include="all")
    missing_counts = thyroid_df.isnull().sum()
    column_types = thyroid_df.dtypes
    return data_summary, missing_counts, column_types

def calculate_jaccard_and_smc(binary_vector1, binary_vector2):
    f11 = f10 = f01 = f00 = 0

    for i in range(len(binary_vector1)):
        if binary_vector1[i] == 1 and binary_vector2[i] == 1:
            f11 += 1
        elif binary_vector1[i] == 1 and binary_vector2[i] == 0:
            f10 += 1
        elif binary_vector1[i] == 0 and binary_vector2[i] == 1:
            f01 += 1
        else:
            f00 += 1

    if (f11 + f10 + f01) == 0:
        jaccard_index = 0
    else:
        jaccard_index = f11 / (f11 + f10 + f01)

    simple_matching_coeff = (f11 + f00) / (f11 + f10 + f01 + f00)

    return jaccard_index, simple_matching_coeff

def calculate_cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    return dot_product / magnitude_product

def plot_similarity_heatmap(input_matrix):
    total_samples = len(input_matrix)
    display_count = min(20, total_samples)

    similarity_matrix = np.zeros((display_count, display_count))

    for i in range(display_count):
        for j in range(display_count):
            numerator = np.dot(input_matrix[i], input_matrix[j])
            denominator = np.linalg.norm(input_matrix[i]) * np.linalg.norm(input_matrix[j])
            similarity_matrix[i][j] = 0 if denominator == 0 else numerator / denominator

    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm")
    plt.title("Cosine Similarity Heatmap")
    plt.show()

def fill_missing_values(data_df):
    for column in data_df.columns:
        if data_df[column].dtype in ["float64", "int64"]:
            data_df[column].fillna(data_df[column].mean(), inplace=True)
        else:
            data_df[column].fillna(data_df[column].mode()[0], inplace=True)
    return data_df

def normalize_data(data_df):
    numeric_cols = data_df.select_dtypes(include=np.number).columns
    for column in numeric_cols:
        min_val = data_df[column].min()
        max_val = data_df[column].max()
        data_df[column] = (data_df[column] - min_val) / (max_val - min_val)
    return data_df

def main():
    excel_path = r"c:\Users\Admin\Documents\Lab Session Data.xlsx"

    input_matrix, payment_vector = load_purchase_data(excel_path)
    print("A1 Vector Space Dimension:", get_vector_space_dimension(input_matrix))
    print("A1 Number of Vectors:", get_number_of_vectors(input_matrix))
    print("A1 Matrix Rank:", calculate_matrix_rank(input_matrix))
    print("A1 Item Cost:\n", calculate_cost_using_pseudo_inverse(input_matrix, payment_vector))

    print("A2 Customer Classification:", classify_customers(payment_vector.flatten()))

    stock_df = load_stock_data(excel_path)
    closing_prices = stock_df.iloc[:, 3].dropna().values

    print("A3 Mean & Variance (NumPy):", mean_variance_numpy(closing_prices))
    print("A3 Mean Time (Manual):", average_time_taken(manual_mean, closing_prices))
    print("A3 Variance Time (Manual):", average_time_taken(manual_variance, closing_prices))
    print("A3 Probability of Loss:", probability_of_loss(stock_df["Chg%"].dropna()))
    print("A3 Probability of Profit on Wednesday:", probability_profit_on_wednesday(stock_df))
    print("A3 Conditional Probability Profit | Wednesday:",
          conditional_probability_profit_wednesday(stock_df))

    plot_change_vs_day(stock_df)

    thyroid_df = load_thyroid_data(excel_path)
    summary, missing, data_types = analyze_thyroid_data(thyroid_df)
    print("A4 Summary:\n", summary)
    print("A4 Missing Values:\n", missing)
    print("A4 Data Types:\n", data_types)

    print("A5 Jaccard & SMC:", calculate_jaccard_and_smc(input_matrix[0], input_matrix[1]))
    print("A6 Cosine Similarity:", calculate_cosine_similarity(input_matrix[0], input_matrix[1]))

    plot_similarity_heatmap(input_matrix)

    thyroid_df = fill_missing_values(thyroid_df)
    thyroid_df = normalize_data(thyroid_df)
    print("A8 & A9 Data Imputation and Normalization Completed")

if __name__ == "__main__":
    main()