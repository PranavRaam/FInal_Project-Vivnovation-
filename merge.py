import pandas as pd

# Load the ZIP-County dataset (ZIP → County mapping)
zip_county_df = pd.read_csv("zip_county.csv")  # Adjust path to your file

# Check the column names in the ZIP-County dataset
print("ZIP-County Dataset Columns:")
print(zip_county_df.columns)

# Clean up any leading/trailing spaces in column names
zip_county_df.columns = zip_county_df.columns.str.strip()

# Rename 'COUNTY' column to 'County FIPS' if necessary
zip_county_df.rename(columns={"COUNTY": "County FIPS"}, inplace=True)

# Load the MSA-County dataset (County → MSA mapping)
msa_county_df = pd.read_csv("msa_data.csv")  # Adjust path to your file

# Check the column names in the MSA-County dataset
print("\nMSA-County Dataset Columns:")
print(msa_county_df.columns)

# Clean up any leading/trailing spaces in column names
msa_county_df.columns = msa_county_df.columns.str.strip()

# Rename 'County Code' column to 'County FIPS' for consistency (if they represent the same thing)
msa_county_df.rename(columns={"County Code": "County FIPS"}, inplace=True)

# Verify the data types of the key columns
print("\nData Types in ZIP-County Dataset:")
print(zip_county_df.dtypes)

print("\nData Types in MSA-County Dataset:")
print(msa_county_df.dtypes)

# Convert the 'County FIPS' column to string if needed
# Ensure both columns are of the same data type (string)
zip_county_df["County FIPS"] = zip_county_df["County FIPS"].astype(str)
msa_county_df["County FIPS"] = msa_county_df["County FIPS"].astype(str)

# Merge the datasets on 'County FIPS'
merged_df = pd.merge(zip_county_df, msa_county_df, on="County FIPS", how="left")

# Inspect the merged dataset
print("\nMerged Dataset:")
print(merged_df.head())

# Save the merged result to a new CSV
merged_df.to_csv("msa_county_zip_merged.csv", index=False)

