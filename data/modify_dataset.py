import pandas as pd
import re

def format_card_number(num, sep):
    # Only format if it's a 16-digit number
    if isinstance(num, str) and re.fullmatch(r"\d{16}", num):
        return f"{num[:4]}{sep}{num[4:8]}{sep}{num[8:12]}{sep}{num[12:]}"
    return num

def modify_card_numbers(input_csv, output_csv):
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    for col in ["credit card number", "debit card number"]:
        # First 100 rows: hyphen
        df.loc[:149, col] = df.loc[:149, col].apply(lambda x: format_card_number(x, "-"))
        # Next 100 rows: space
        df.loc[150:299, col] = df.loc[150:299, col].apply(lambda x: format_card_number(x, " "))

    df.to_csv(output_csv, index=False)
    print(f"Card numbers formatted and saved to {output_csv}")

if __name__ == "__main__":
    modify_card_numbers("pii_synthetic_dataset_gpt2_labeled.csv", "pii_synthetic_dataset_gpt2_labeled_modified.csv")