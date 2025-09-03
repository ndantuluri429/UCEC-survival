from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import os
import zipfile
import io
import pandas as pd
import shutil
import tempfile
import json
import polars as pl
import numpy as np

methylation_cancer_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/methylation_df_filtered.parquet')

"""### LOAD IN FILE"""

#load in the files needed
clinical_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/clinical_df_polars.parquet')
methylation_norm_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/methylation_df_norm_filtered.parquet')
#methylation_cancer_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/methylation_df_filtered.parquet')
methylation_cancer_normal_match_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/methylation_df_filetered_cancer_match.parquet')
cnv_df = pl.read_parquet('/content/drive/MyDrive/bmi702_final_project/intermediate_data_from_exploration/cnv_amplified_deleted_df.parquet')

#sanity check of seeing how many unique_case_ids each has
print("Clinical df case ids: ", clinical_df['case_id'].n_unique())
print("Methylation cancer df case ids: ", methylation_cancer_df['case_id'].n_unique())
print("Methylation cancer normal match df case ids: ", methylation_cancer_normal_match_df['case_id'].n_unique())
print("CNV df case ids: ", cnv_df['case_id'].n_unique())
print("Methylatio normal: ", methylation_norm_df['case_id'].n_unique())

"""### Create DF with random genes

### CONDUCT P VALUE TEST AND IDENTIFY TOP 10 METHYLATION
"""

methylation_cancer_normal_match_df.shape

methylation_norm_df.shape

# Start with just the data you need
cancer_pivot = (
    methylation_cancer_normal_match_df
    .select(["gene_name", "case_id", "beta_value"])
    .pivot(values="beta_value", index="gene_name", columns="case_id", aggregate_function="first")
)

normal_pivot = (
    methylation_norm_df
    .select(["gene_name", "case_id", "beta_value"])
    .pivot(values="beta_value", index="gene_name", columns="case_id", aggregate_function="first")
)

common_cases = set(cancer_pivot.columns[1:]).intersection(set(normal_pivot.columns[1:]))
common_cases = sorted(common_cases)  # sort to align columns

from statsmodels.stats import multitest
from scipy.stats import wilcoxon

pvals = []

for row_idx in range(cancer_pivot.height):
    gene = cancer_pivot[row_idx, 0]

    # Skip bad gene names
    if gene in (None, "NA", "", "NaN"):
        continue

    # Get cancer and normal beta values
    cancer_vals = cancer_pivot[row_idx, 1:].to_numpy()[0]
    normal_match = normal_pivot.filter(pl.col("gene_name") == gene)

    if normal_match.is_empty():
        continue

    normal_vals = normal_match.row(0)[1:]

    # Get values only for common case_ids
    cancer_common = [
        cancer_vals[cancer_pivot.columns.index(col) - 1] for col in common_cases
    ]
    normal_common = [
        normal_vals[normal_pivot.columns.index(col) - 1] for col in common_cases
    ]

    # Convert to numpy arrays
    cancer_arr = np.array(cancer_common, dtype=np.float64)
    normal_arr = np.array(normal_common, dtype=np.float64)

    # Drop NaNs
    mask = ~np.isnan(cancer_arr) & ~np.isnan(normal_arr)
    cancer_arr = cancer_arr[mask]
    normal_arr = normal_arr[mask]

    if len(cancer_arr) < 3 or np.allclose(cancer_arr, normal_arr):
        pvals.append((gene, 1.0))
        continue

    try:
        _, p_val = wilcoxon(cancer_arr, normal_arr)
        pvals.append((gene, p_val))
    except Exception as e:
        print(f"Skipping {gene}: {e}")
        continue

# If nothing passed, avoid crash
if not pvals:
    print("No valid genes for testing.")
else:
    # Step 4: Multiple testing correction (Benjamini-Hochberg)
    genes, raw_pvals = zip(*pvals)
    _, corrected_pvals, _, _ = multitest.multipletests(raw_pvals, method="fdr_bh")

    # Step 5: Create final DataFrame
    pval_df = pl.DataFrame({
        "gene_name": genes,
        "p_value": raw_pvals,
        "corrected_p_value": corrected_pvals
    })

    # Step 6: Filter significant genes (adjusted p < 0.05)
    significant_genes_df = pval_df.filter(pl.col("corrected_p_value") < 0.05)

    # Output
    print("Top significant genes:")
    print(significant_genes_df.sort("corrected_p_value").head(10))

# Calculate the median beta values for normal and cancer
methylated_df_norm_avg = methylation_norm_df.group_by("gene_name").agg([
    pl.col("beta_value").median().alias("median_beta_value_normal"),  # Median for normal
    pl.col("case_id").unique().alias("case_ids_normal")  # Unique case_ids for normal
])

methylation_df_filtered_avg = methylation_cancer_normal_match_df.group_by("gene_name").agg([
    pl.col("beta_value").median().alias("median_beta_value_cancer"),  # Median for cancer
    pl.col("case_id").unique().alias("case_ids_cancer")  # Unique case_ids for cancer
])

# Merge cancer and normal dataframes on gene_name
merged_norm_cancer_df = methylation_df_filtered_avg.join(
    methylated_df_norm_avg,
    on="gene_name",
    how="inner"
)

# 2. Now, let's **merge** the significant genes (those you identified from the pairwise tests) with the median values
merged_significant_genes = significant_genes_df.join(
    merged_norm_cancer_df,
    on="gene_name",
    how="left"
)

# 3. **Calculate the effect size** as the absolute difference between cancer and normal median beta values
merged_significant_genes = merged_significant_genes.with_columns(
    (pl.col("median_beta_value_cancer") - pl.col("median_beta_value_normal")).abs().alias("effect_size")
)

# Display the merged dataframe with effect size
print("Significant genes with effect size:")
print(merged_significant_genes)

## 1. Sort by the absolute effect size in descending order
merged_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")  # Take the absolute value of effect_size
).sort("abs_effect_size", descending=True)

# 2. Drop unnecessary columns
merged_significant_genes = merged_significant_genes.drop(
    ["abs_effect_size", "case_ids_cancer", "case_ids_normal"]
)

# 3. Convert to Pandas DataFrame and rename the columns as needed
merged_significant_genes_pandas = merged_significant_genes.head().to_pandas()
merged_significant_genes_pandas.columns = ["gene_name", "p_value", "corrected_p_value",
                                           "median_beta_value_cancer", "median_beta_value_normal", "effect_size"]

# Display the top genes (top 10 for example)
print("Top significant genes:")
print(merged_significant_genes_pandas)

#plot the top 10 most methylated via effect size
import matplotlib.pyplot as plt

# Sort the DataFrame by the 'effect_size' column in descending order
merged_significant_genes_plot  = merged_significant_genes.sort("effect_size", descending=True).head(10)

#now plot
plt.figure(figsize=(10, 6))
plt.bar(merged_significant_genes_plot["gene_name"], merged_significant_genes_plot["effect_size"], color='skyblue')
plt.xlabel('Gene Name')
plt.ylabel('Effect Size')
plt.title('Top 10 Most Methylated Genes by Effect Size')

# extract out various dataframes of top 10 methylated genes, top 1000 methylated genes, and top 30
top_1000_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")  # Take absolute value of effect_size
).sort("abs_effect_size", descending=True).head(1000)

top_10_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")  # Take absolute value of effect_size
).sort("abs_effect_size", descending=True).head(10)

top_30_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")  # Take absolute value of effect_size
).sort("abs_effect_size", descending=True).head(30)

# 5. Match these top genes to the methylation_cancer_df
methylation_top_1000_df = methylation_cancer_df.filter(
    pl.col("gene_name").is_in(top_1000_significant_genes["gene_name"])
)

methylation_top_10_df = methylation_cancer_df.filter(
    pl.col("gene_name").is_in(top_10_significant_genes["gene_name"])
)

methylation_top_30_df = methylation_cancer_df.filter(
    pl.col("gene_name").is_in(top_30_significant_genes["gene_name"])
)

#random list of genes
random_genes_25 = [
    "RPLP0", "ACTB", "PPP1CA",'PRPH2','SLC26A5']

random_genes = methylation_cancer_df.filter(
    pl.col("gene_name").is_in(random_genes_25)
)

#NOTE: they are not in any of these genes, so now for the top 30 we create a new df with them added
random_genes['gene_name'].unique()

#check if random_genes_25 are in any of the methylation_top dfs

#creating a new methylation_top_30_df that has these random genes as well
methylation_top_30_invariant_df = methylation_top_30_df.vstack(random_genes)
methylation_top_10_invariant_df = methylation_top_10_df.vstack(random_genes)

#sanity check to see how many unique genes each df has:
print("Methylation top 1000 df case ids: ", methylation_top_1000_df['gene_name'].n_unique())
print("Methylation top 10 df case ids: ", methylation_top_10_df['gene_name'].n_unique())
print("Methylation top 30 df case ids: ", methylation_top_30_df['gene_name'].n_unique())
print("Methylation top 30 invariant df case ids: ", methylation_top_30_invariant_df['gene_name'].n_unique())
print("Methylation top 10 invariant df case ids: ", methylation_top_10_invariant_df['gene_name'].n_unique())

"""### TRYING TO SEE IF WE HAVE TO IMPUTE METHYLATION"""

collapsed_df_1000 = (
   methylation_top_1000_df
    .group_by(["case_id", "gene_name"])
    .agg(pl.col("beta_value").mean().alias("beta_value"))
)

collapsed_df_10 = (
   methylation_top_10_df
    .group_by(["case_id", "gene_name"])
    .agg(pl.col("beta_value").mean().alias("beta_value"))
)

collapsed_df_30 = (
   methylation_top_30_df
    .group_by(["case_id", "gene_name"])
    .agg(pl.col("beta_value").mean().alias("beta_value"))
)

collapsed_df_30_invariant = (
   methylation_top_30_invariant_df
    .group_by(["case_id", "gene_name"])
    .agg(pl.col("beta_value").mean().alias("beta_value"))
)

collapsed_df_10_invariant = (
    methylation_top_10_invariant_df
    .group_by(["case_id", "gene_name"])
    .agg(pl.col("beta_value").mean().alias("beta_value"))
)

collapsed_df_30_invariant['gene_name'].n_unique()

collapsed_df_10_invariant['gene_name'].n_unique()

collapsed_df_30['gene_name'].n_unique()

#unique_case_ids = methylation_top_10_df.select("case_id").unique().to_series().to_list()
#results = []

# Process in batches (you can adjust batch size if needed)
#batch_size = 100
#for i in range(0, len(unique_case_ids), batch_size):
#    batch_case_ids = unique_case_ids[i:i+batch_size]

    # Filter to the current batch
#    batch_df = methylation_top_10_df.filter(pl.col("case_id").is_in(batch_case_ids))

    # Group and aggregate within this batch
#    collapsed_batch = (
#        batch_df
#        .group_by(["case_id", "gene_name"])
 #       .agg(pl.col("beta_value").mean().alias("beta_value"))
#    )

 #   results.append(collapsed_batch)

# Combine all the batches
#collapsed_df = pl.concat(results, how="vertical")

"""### IMPUTING METHYLATION WITH JUST MEDIAN VALUE"""

def make_filled_meth_df(collapsed_df):
  # Unique lists
  unique_cases = collapsed_df.select("case_id").unique()
  unique_genes = collapsed_df.select("gene_name").unique()

  # Cartesian product (full grid)
  full_grid = unique_cases.join(unique_genes, how="cross")

  # Left join to the collapsed_df
  filled_df = full_grid.join(collapsed_df, on=["case_id", "gene_name"], how="left")

  # Gene-level mean imputation
  gene_means = (
      collapsed_df
      .group_by("gene_name")
      .agg(pl.col("beta_value").mean().alias("gene_mean"))
  )

  filled_meth_df = filled_df.join(gene_means, on="gene_name", how="left")

  filled_meth_df = filled_meth_df.with_columns(
      pl.when(pl.col("beta_value").is_null())
      .then(pl.col("gene_mean"))
      .otherwise(pl.col("beta_value"))
      .alias("beta_value")
  ).drop("gene_mean")

  return filled_meth_df

#making filed_meth_df for all collapsed
filled_meth_df_1000 = make_filled_meth_df(collapsed_df_1000)
filled_meth_df_10 = make_filled_meth_df(collapsed_df_10)
filled_meth_df_30 = make_filled_meth_df(collapsed_df_30)
filled_meth_df_30_invariant = make_filled_meth_df(collapsed_df_30_invariant)
filled_meth_df_10_invariant = make_filled_meth_df(collapsed_df_10_invariant)

filled_meth_df_30['gene_name'].n_unique()

#creating pivoted dfs
pivoted_meth_df_1000 = filled_meth_df_1000.pivot(
    values="beta_value",
    index="case_id",
    on="gene_name",
    aggregate_function="first"
)

pivoted_meth_df_10 = filled_meth_df_10.pivot(
    values="beta_value",
    index="case_id",
    on="gene_name",
    aggregate_function="first"
)

pivoted_meth_df_30 = filled_meth_df_30.pivot(
    values="beta_value",
    index="case_id",
    on="gene_name",
    aggregate_function="first"
)

pivoted_meth_df_30_invariant = filled_meth_df_30_invariant.pivot(
    values="beta_value",
    index="case_id",
    on="gene_name",
    aggregate_function="first"
)

pivoted_meth_df_10_invariant = filled_meth_df_10_invariant.pivot(
    values="beta_value",
    index="case_id",
    on="gene_name",
    aggregate_function="first"
)

"""### CREATE GENE AMPLIFICATION AND GENE DEPLETION COLUMNS"""

cnv_with_flags = cnv_df.with_columns([
    (pl.col("copy_number") > 2).cast(pl.Float64).alias("is_amplified"),
    (pl.col("copy_number") < 2).cast(pl.Float64).alias("is_depleted")
])

# Group by case_id and take the mean
cnv_summary_avg = cnv_with_flags.group_by("case_id").agg([
    pl.col("is_amplified").mean().alias("gene_amplification"),
    pl.col("is_depleted").mean().alias("gene_depletion"),
    pl.col("gene_name").n_unique().alias("unique_genes"),
    pl.count().alias("total_gene_entries")
])

# Final output
final_df_cnv = cnv_summary_avg.select(["case_id", "gene_amplification", "gene_depletion"])

"""## COMBINING ALL INTO ONE GENOMIC DF"""

#checking how many unique case_ids exist for methylation and cnv_df
print("Methylation df case ids: ", pivoted_meth_df_1000['case_id'].n_unique())
print("CNV df case ids: ", final_df_cnv['case_id'].n_unique())

#combining methylation and cnv based on case_ids
combined_genomic_df_1000 = pivoted_meth_df_1000.join(final_df_cnv, on="case_id", how="inner")
combined_genomic_df_10 = pivoted_meth_df_10.join(final_df_cnv, on="case_id", how="inner")
combined_genomic_df_30 = pivoted_meth_df_30.join(final_df_cnv, on="case_id", how="inner")
combined_genomic_df_30_invariant = pivoted_meth_df_30_invariant.join(final_df_cnv, on="case_id", how="inner")
combined_genomic_df_10_invariant = pivoted_meth_df_10_invariant.join(final_df_cnv, on="case_id", how="inner")

#save off the list of genes (ie cols that are not case_id, gene_amplification, gene_depletion)
# Columns to exclude
exclude = {"case_id", "gene_amplification", "gene_depletion"}

#get a list of the columns
gene_cols_10000 = combined_genomic_df_1000.columns
gene_cols_10 = combined_genomic_df_10.columns
gene_cols_30 = combined_genomic_df_30.columns
gene_cols_30_invariant = combined_genomic_df_30_invariant.columns
gene_cols_10_invariant = combined_genomic_df_10_invariant.columns

#create a cleaned list of columns
gene_cols_10000 = [col for col in gene_cols_10000 if col not in exclude]
gene_cols_10 = [col for col in gene_cols_10 if col not in exclude]
gene_cols_30 = [col for col in gene_cols_30 if col not in exclude]
gene_cols_30_invariant = [col for col in gene_cols_30_invariant if col not in exclude]
gene_cols_10_invariant = [col for col in gene_cols_10_invariant if col not in exclude]

##sanity check print the len of each column list
print(len(gene_cols_10000))
print(len(gene_cols_10))
print(len(gene_cols_30))
print(len(gene_cols_30_invariant))
print(len(gene_cols_10_invariant))

"""### Data imputation and normalization for clinical_df"""

#convert clinical df back to pandas
clinical_df = clinical_df.to_pandas()

columns = clinical_df.columns.values.tolist()

# finding missing vals
missing_values = clinical_df.isnull().sum()
missing_dash = (clinical_df == "'--").sum()
combined_missing = missing_values + missing_dash
cols_with_missing = combined_missing[combined_missing > 0]
cols_with_missing.head()

# df info
unique_values = clinical_df['case_id'].unique()
print(len(unique_values))

#aggregating case ids to have 1 row per patient
def aggregate_column(series):
    filtered_series = series[~((series == "'--") | (series.str.lower() == 'not reported'))] # do not include their null char or not reported
    unique_values = filtered_series.unique()

    if len(unique_values) == 1:
        return unique_values[0]  # Return the value if all are the same
    elif len(unique_values) == 0:
      return "'--"
    else:
        return "; ".join(map(str, filtered_series))  # Return a semicolon seperated string if there are multiple values

clinical_df_aggregated = clinical_df.groupby('case_id').agg(aggregate_column).reset_index()

# remove rows with missingness above 500
missing_values = clinical_df_aggregated.isnull().sum()
missing_dash = (clinical_df_aggregated == "'--").sum()
combined_missing = missing_values + missing_dash
cols_to_remove = combined_missing[combined_missing>500].index
print(cols_to_remove)
clinical_df_aggregated = clinical_df_aggregated.drop(columns=cols_to_remove)

# remove rows containing "'--" in the target var and time var
clinical_df_aggregated = clinical_df_aggregated[clinical_df_aggregated['vital_status'] != "'--"]
clinical_df_aggregated = clinical_df_aggregated[clinical_df_aggregated['days_to_birth'] != "'--"]

# remove unneccessary cols
cols_to_remove = ['age_is_obfuscated', 'country_of_residence_at_enrollment', 'figo_staging_edition_year', 'clinical_trial_indicator',
                  'course_number', 'number_of_cycles', 'number_of_fractions', 'residual_disease.1', 'treatment_intent_type', 'treatment_outcome',
                  'days_to_treatment_end', 'days_to_treatment_start','file_name']
clinical_df_aggregated = clinical_df_aggregated.drop(columns=cols_to_remove)

# replace all '-- with None
clinical_df_aggregated = clinical_df_aggregated.replace("'--", None)
# inferring missing values

# can be = 0
cols_to_0 = ['treatment_dose', 'prescribed_dose']
clinical_df_aggregated[cols_to_0] = clinical_df_aggregated[cols_to_0].fillna(0)

# can be Unknown
cols_to_uk = ['tumor_of_origin', 'residual_disease', 'synchronous_malignancy', 'prior_malignancy', 'method_of_diagnosis', 'race', 'ethnicity', 'year_of_diagnosis']
clinical_df_aggregated[cols_to_uk] = clinical_df_aggregated[cols_to_uk].fillna('Unknown')

# units
cols_to_mg = ['treatment_dose_units', 'prescribed_dose_units']
clinical_df_aggregated[cols_to_mg] = clinical_df_aggregated[cols_to_mg].fillna('mg')

# all cols with ;
cols_with_semicolon = [col for col in clinical_df_aggregated.columns if clinical_df_aggregated[col].astype(str).str.contains(";").any()]
print(cols_with_semicolon)
len(cols_with_semicolon)

# collapsing further from semi-colon sep to a single value

cols_with_multi = ['age_at_diagnosis', 'classification_of_tumor', 'days_to_diagnosis', 'diagnosis_is_primary_disease', 'figo_stage',
                   'morphology', 'primary_diagnosis', 'prior_treatment', 'residual_disease', 'tissue_or_organ_of_origin',
                   'initial_disease_status', 'prescribed_dose', 'prescribed_dose_units', 'therapeutic_agents', 'treatment_dose',
                   'treatment_dose_units', 'treatment_or_therapy', 'treatment_type']

# columns to make binary from presence of yes
col_yes = ['prior_treatment', 'treatment_or_therapy']
for col in col_yes:
  clinical_df_aggregated[col] = clinical_df_aggregated[col].apply(lambda x: 'yes' if 'yes' in x.lower().split('; ') else 'no')

# replace days_to_diagnosis column with highest val
clinical_df_aggregated['days_to_diagnosis'] = clinical_df_aggregated['days_to_diagnosis'].apply(
    lambda value: max(map(int, value.split('; '))) if isinstance(value, str) else value)


# replace certain cols with the latest value
col_latest = ['age_at_diagnosis', 'classification_of_tumor', 'days_to_diagnosis', 'diagnosis_is_primary_disease', 'figo_stage', 'morphology',
              'primary_diagnosis', 'residual_disease', 'initial_disease_status', 'prescribed_dose', 'prescribed_dose_units', 'therapeutic_agents',
              'treatment_dose', 'treatment_dose_units']
for col in col_latest:
  clinical_df_aggregated[col] = clinical_df_aggregated[col].apply(lambda x: str(x).split('; ')[0] if isinstance(x, str) else x)

# treatment type --> add up number of unique treatments patient had
clinical_df_aggregated['treatment_type_count'] = clinical_df_aggregated['treatment_type'].apply(lambda x: len(set(x.split('; '))))

# tissue_or_organ_of_origin -> set of tissues
clinical_df_aggregated['tissue_or_organ_of_origin'] = clinical_df_aggregated['tissue_or_organ_of_origin'].apply(lambda x: '; '.join(sorted(set(x.split('; ')))))
clinical_df_aggregated['tissue_or_organ_of_origin_count'] = clinical_df_aggregated['tissue_or_organ_of_origin'].apply(lambda x: len((set(x.split('; ')))))

# making morphology into 2 cols
clinical_df_aggregated[['histology_code', 'behavior_code']] = clinical_df_aggregated['morphology'].str.split('/', expand=True)

# add time to event col
# days to death for those alive = latest days to diagnosis (highest number in days to diag)
clinical_df_aggregated['time_to_event'] = clinical_df_aggregated['days_to_death'].fillna(clinical_df_aggregated['days_to_diagnosis'])

# cols to drop --> inferred, useless or used
cols_to_drop = ['treatment_type', 'morphology', 'days_to_death', 'days_to_diagnosis',
                'timepoint_category', 'treatment_or_therapy', 'gender', 'tumor_of_origin', 'year_of_diagnosis']
clinical_df_aggregated = clinical_df_aggregated.drop(columns=cols_to_drop)

# final missingness check
clinical_df_aggregated.isna().sum()

# printing unique values in cols to build encoding dict
for col in clinical_df_aggregated.columns:
  if col in ['case_id', 'case_submitter_id', 'project_id']:
    continue;
  else:
    print(col, list(clinical_df_aggregated[col].unique()))

clinical_df_aggregated.columns

# create encoded df
clinical_df_encoded = clinical_df_aggregated.copy()

# encoding dictionaries for binary and ordinal variables
diagnosis_is_primary_disease = {'true': 1, 'false':0}
prior_treatment = {'yes': 1, 'no':0}
prior_malignancy = {'yes': 1, 'no':0, 'Unknown':0}
synchronous_malignancy = {'Yes': 1, 'No':0, 'Unknown':0}
vital_status = {'Alive': 0, 'Dead': 1}
classification_of_tumor = {'Prior primary':0, 'primary':1, 'Subsequent Primary':2, 'Synchronous primary':3, 'recurrence':4, 'metastasis':5}
figo_stage = {'Stage IA':0,
    'Stage IB1':0,
    'Stage IB':0,
    'Stage IC':0,
    'Stage I':0,
    'Stage IIA':1,
    'Stage IIB':1,
    'Stage II':1,
    'Stage IIIA':2,
    'Stage IIIB':2,
    'Stage IIIC1':2,
    'Stage IIIC2':2,
    'Stage IIIC':2,
    'Stage III':2,
    'Stage IVA':3,
    'Stage IVB':3,
    'Stage IV':3}
site_of_resection_or_biopsy = {'Endometrium':0, 'Fundus uteri':1}
tumor_grade = {'G1':0, 'G2':1, 'G3':2, 'High Grade':3}
residual_disease = {'Unknown':0, 'R0':0, 'R1':1, 'R2':2, 'RX':3}
initial_disease_status = {'Initial Diagnosis':0, 'Progressive Disease':1, 'Recurrent Disease':2, 'Persistent Disease':3}

# mapping encoding to value
clinical_df_encoded['diagnosis_is_primary_disease'] = clinical_df_encoded['diagnosis_is_primary_disease'].map(diagnosis_is_primary_disease)
clinical_df_encoded['prior_treatment'] = clinical_df_encoded['prior_treatment'].map(prior_treatment)
clinical_df_encoded['prior_malignancy'] = clinical_df_encoded['prior_malignancy'].map(prior_malignancy)
clinical_df_encoded['synchronous_malignancy'] = clinical_df_encoded['synchronous_malignancy'].map(synchronous_malignancy)
clinical_df_encoded['vital_status'] = clinical_df_encoded['vital_status'].map(vital_status)
clinical_df_encoded['classification_of_tumor'] = clinical_df_encoded['classification_of_tumor'].map(classification_of_tumor)
clinical_df_encoded['figo_stage'] = clinical_df_encoded['figo_stage'].map(figo_stage)
clinical_df_encoded['site_of_resection_or_biopsy'] = clinical_df_encoded['site_of_resection_or_biopsy'].map(site_of_resection_or_biopsy)
clinical_df_encoded['tumor_grade'] = clinical_df_encoded['tumor_grade'].map(tumor_grade)
clinical_df_encoded['residual_disease'] = clinical_df_encoded['residual_disease'].map(residual_disease)
clinical_df_encoded['initial_disease_status'] = clinical_df_encoded['initial_disease_status'].map(initial_disease_status)

#numeric cols
cols_numeric = ['age_at_index', 'days_to_birth', 'age_at_diagnosis',
                'time_to_event', 'histology_code', 'behavior_code',
                'treatment_type_count', 'tissue_or_organ_of_origin_count',
                'treatment_dose']

for col in cols_numeric:
    clinical_df_encoded[col] = pd.to_numeric(clinical_df_encoded[col], errors='coerce')

# prescribed_dose : str -> flt -> int
clinical_df_encoded['prescribed_dose'] = clinical_df_encoded['prescribed_dose'].astype(float)
clinical_df_encoded['prescribed_dose'] = clinical_df_encoded['prescribed_dose'].astype(int)

# one-hot endocing of remaning cols
cols_one_hot = ['icd_10_code', 'method_of_diagnosis', 'primary_diagnosis', 'prescribed_dose_units',
                'therapeutic_agents', 'treatment_dose_units', 'ethnicity', 'race']
clinical_df_encoded = pd.get_dummies(clinical_df_encoded, columns=cols_one_hot, drop_first=True, dtype=int)

# tissue_or_organ_of_origin one-hot encoding
tissues = clinical_df_encoded['tissue_or_organ_of_origin'].str.get_dummies(sep='; ')
tissues.columns = [f'tissue_or_organ_of_origin {col}' for col in tissues.columns]
clinical_df_encoded = pd.concat([clinical_df_encoded, tissues.iloc[:, 1:]], axis=1)
clinical_df_encoded = clinical_df_encoded.drop(columns=['tissue_or_organ_of_origin'])

# checking that all cols are encoded and numeric
non_numeric_cols = clinical_df_encoded.select_dtypes(exclude=['int64']).columns
non_numeric_cols

# add +1 to time_to_event to avoid 0
clinical_df_encoded['time_to_event'] = clinical_df_encoded['time_to_event'] + 1

# correlation
df = clinical_df_encoded.iloc[:, 3:]
correlation_matrix = df.corr()

# find highly correlaed cols
threshold = 0.8
mask = np.triu(np.ones(correlation_matrix.shape), k=1)
high_corr = correlation_matrix.where(mask > 0).stack().reset_index()
high_corr.columns = ['Variable_1', 'Variable_2', 'Correlation']
high_corr_filtered = high_corr[high_corr['Correlation'].abs() > threshold]
high_corr_filtered

# columns to remove (correlation high > 0.8)
cols_to_remove = ['days_to_birth', 'age_at_diagnosis', 'diagnosis_is_primary_disease', 'primary_diagnosis_Pheochromocytoma, NOS',
                  'tissue_or_organ_of_origin Endometrium', 'tissue_or_organ_of_origin Fundus uteri', 'tissue_or_organ_of_origin Corpus uteri',
                  'tissue_or_organ_of_origin Kidney, NOS', 'primary_diagnosis_Serous cystadenocarcinoma, NOS', 'tissue_or_organ_of_origin Blood',
                  'tissue_or_organ_of_origin Cecum', 'therapeutic_agents_Cyclophosphamide', 'treatment_dose_units_cGy']

clinical_df_encoded = clinical_df_encoded.drop(columns=cols_to_remove)

# remove features with very low variance < 0.05
df = clinical_df_encoded.drop(columns=['case_id', 'project_id', 'case_submitter_id', 'vital_status', 'time_to_event'])
feature_variance = df.var()
threshold = 0.05
drop_low_var = feature_variance[feature_variance < threshold].index
clinical_df_encoded = clinical_df_encoded.drop(columns=drop_low_var)

# how many features remain
clinical_df_encoded.shape

#create combined genomic clinical pd dfs and save off
combined_genomic_df_1000_pandas = combined_genomic_df_1000.to_pandas()
combined_genomic_df_1000.write_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_df_1000.csv')

combined_genomic_df_10_pandas = combined_genomic_df_10.to_pandas()
combined_genomic_df_10.write_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_df_10.csv')

combined_genomic_df_30_pandas = combined_genomic_df_30.to_pandas()
combined_genomic_df_30.write_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_df_30.csv')

combined_genomic_df_30_invariant_pandas = combined_genomic_df_30_invariant.to_pandas()
combined_genomic_df_30_invariant.write_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_df_30_invariant.csv')

combined_genomic_df_10_invariant_pandas = combined_genomic_df_10_invariant.to_pandas()
combined_genomic_df_10_invariant.write_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_df_10_invariant.csv')

#combining genomic with clinical df
#combined_genomic_df_pandas = combined_genomic_df.to_pandas()
#combined_genomic_df.write_csv('combined_genomic_ALL_df.csv')

#combined_genomic_df_pandas.columns

#combining each genomic with clinical df
#combined_genomic_df_1000_pandas['case_id'] = combined_genomic_df_1000_pandas['case_id'].astype(str)
#clinical_df_encoded['case_id'] = clinical_df_encoded['case_id'].astype(str)

combined_genomic_clinical_df_1000 = pd.merge(
    combined_genomic_df_1000_pandas,
    clinical_df_encoded,
    on='case_id',
    how='inner'
)

combined_genomic_clinical_df_10 = pd.merge(
    combined_genomic_df_10_pandas,
    clinical_df_encoded,
    on='case_id',
    how='inner'
)

combined_genomic_clinical_df_30 = pd.merge(
    combined_genomic_df_30_pandas,
    clinical_df_encoded,
    on='case_id',
    how='inner'
)

combined_genomic_clinical_df_30_invariant = pd.merge(
    combined_genomic_df_30_invariant_pandas,
    clinical_df_encoded,
    on='case_id',
    how='inner'
)

combined_genomic_clinical_df_10_invariant = pd.merge(
    combined_genomic_df_10_invariant_pandas,
    clinical_df_encoded,
    on='case_id',
    how='inner'
)

#combining genomic with clinical df
#combined_genomic_df_pandas['case_id'] = combined_genomic_df_pandas['case_id'].astype(str)
#clinical_df_encoded['case_id'] = clinical_df_encoded['case_id'].astype(str)
#combined_genomic_clinical_df = pd.merge(
#    combined_genomic_df_pandas,
#    clinical_df_encoded,
#    on='case_id',
#    how='inner'
#)

#saving off these combined genomic_clinical dfs
combined_genomic_clinical_df_1000.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_clinical_df_1000.csv', index=False)
combined_genomic_clinical_df_10.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_clinical_df_10.csv', index=False)
combined_genomic_clinical_df_30.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_clinical_df_30.csv', index=False)
combined_genomic_clinical_df_30_invariant.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_clinical_df_30_invariant.csv', index=False)
combined_genomic_clinical_df_10_invariant.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/combined_genomic_clinical_df_10_invariant.csv', index=False)

def scaling_continous(cols_list, combined_df):
  from sklearn.preprocessing import StandardScaler
  columns_no_scale = ['case_id', 'project_id', 'case_submitter_id', 'vital_status', 'time_to_event']
  continuous_cols = ['age_at_index', 'prescribed_dose', 'treatment_dose']
  discrete_cold = ['tissue_or_organ_of_origin_count', 'treatment_type_count']
  ordinal_cols = ['classification_of_tumor', 'figo_stage', 'residual_disease', 'tumor_grade', 'initial_disease_status']
  binary_cols = ['prior_malignancy', 'prior_treatment', 'histology_code',
        'method_of_diagnosis_Dilation and Curettage Procedure',
        'method_of_diagnosis_Surgical Resection',
        'primary_diagnosis_Endometrioid adenocarcinoma, NOS',
        'prescribed_dose_units_mg', 'therapeutic_agents_Carboplatin',
        'therapeutic_agents_Paclitaxel', 'therapeutic_agents_Tamoxifen',
        'treatment_dose_units_mg', 'ethnicity_not hispanic or latino',
        'race_black or african american', 'race_white']

  appended_continous_cols = continuous_cols + cols_list
  print(appended_continous_cols)
  # z-scale only continuous and ordinal columns
  #print(combined_df.columns)
  df = combined_df.drop(columns=['case_id', 'project_id', 'case_submitter_id'])
  scaler = StandardScaler()
  df[appended_continous_cols] = scaler.fit_transform(df[appended_continous_cols])

  return df

#creating scaled dfs
scaled_df_1000 = scaling_continous(gene_cols_10000, combined_genomic_clinical_df_1000)
scaled_df_10 = scaling_continous(gene_cols_10, combined_genomic_clinical_df_10)
scaled_df_30 = scaling_continous(gene_cols_30, combined_genomic_clinical_df_30)
scaled_df_30_invariant = scaling_continous(gene_cols_30_invariant, combined_genomic_clinical_df_30_invariant)
scaled_df_10_invariant = scaling_continous(gene_cols_10_invariant, combined_genomic_clinical_df_10_invariant)

#saving off the scaled dfs
scaled_df_1000.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/scaled_df_1000.csv', index=False)
scaled_df_10.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/scaled_df_10.csv', index=False)
scaled_df_30.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/scaled_df_30.csv', index=False)
scaled_df_30_invariant.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/scaled_df_30_invariant.csv', index=False)
scaled_df_10_invariant.to_csv('/content/drive/MyDrive/bmi702_final_project/final_dataframes_for_models/updated_datasets/scaled_df_10_invariant.csv', index=False)

"""### WRONG P -TEST CALUCLATIONS BELOW"""

#conduct p-test to determine which are the top 10 most significant genes

#find average beta_values for normal methylation
methylated_df_norm_avg = methylation_norm_df.group_by("gene_name").agg([
    pl.col("beta_value").median().alias("median_beta_value"),  # Calculate the mean of beta values
    pl.col("case_id").unique().alias("case_ids")  # Keep the unique case_ids for each gene
])

# Display the first few rows
print(methylated_df_norm_avg.head())

#get average beta values for cancer methylation
methylation_df_filtered_avg = methylation_cancer_normal_match_df.group_by("gene_name").agg([
    pl.col("beta_value").median().alias("median_beta_value"),  # Calculate the mean of beta values
    pl.col("case_id").unique().alias("case_ids")  # Keep the unique case_ids for each gene
])

# Display the first few rows
print(methylation_df_filtered_avg.head())

merged_norm_cancer_df = methylation_df_filtered_avg.join(methylated_df_norm_avg, on="gene_name", how="inner", suffix="_normal")
merged_norm_cancer_df['gene_name'].n_unique()

#actually calculating p-values using wicoxon signed rank (paired t-test) and BH correction
from scipy.stats import wilcoxon
from statsmodels.stats import multitest
import polars as pl
import numpy as np

# Initialize a list to store the results (gene_name, p_value)
p_values = []

# Iterate through each row (gene) in the merged DataFrame
for row in merged_norm_cancer_df.iter_rows():
    gene_name = row[0]
    cancer_case_ids = row[2]  # Case IDs for cancer samples
    normal_case_ids = row[4]  # Case IDs for normal samples
    cancer_beta_values = [row[1]] * len(cancer_case_ids)  # Beta values for cancer samples
    normal_beta_values = [row[3]] * len(normal_case_ids)  # Beta values for normal samples

    # Create a dictionary to map case_ids to their corresponding beta values for cancer and normal datasets
    cancer_case_map = dict(zip(cancer_case_ids, cancer_beta_values))
    normal_case_map = dict(zip(normal_case_ids, normal_beta_values))

    # Find the intersection of case_ids (patients that appear in both datasets)
    common_case_ids = set(cancer_case_map.keys()).intersection(normal_case_map.keys())

    # If there are common patients, we can proceed with the Wilcoxon test
    if common_case_ids:
        cancer_values = [cancer_case_map[case_id] for case_id in common_case_ids]
        normal_values = [normal_case_map[case_id] for case_id in common_case_ids]

        # Convert to NumPy arrays for compatibility with np.isnan and np.isinf
        cancer_values = np.array(cancer_values, dtype=np.float64)  # Ensure it's a float array
        normal_values = np.array(normal_values, dtype=np.float64)

        # Replace None with np.nan for both arrays (in case there are still None values)
        cancer_values = np.where(np.isnan(cancer_values), np.nan, cancer_values)
        normal_values = np.where(np.isnan(normal_values), np.nan, normal_values)

        # Check if any NaN or infinite values are present in the cancer or normal values
        if np.any(np.isnan(cancer_values)) or np.any(np.isnan(normal_values)):
            print(f"Warning: NaN values found for gene {gene_name}. Skipping this gene.")
            continue

        if np.any(np.isinf(cancer_values)) or np.any(np.isinf(normal_values)):
            print(f"Warning: Infinite values found for gene {gene_name}. Skipping this gene.")
            continue

        # Check if the differences between cancer and normal values are all zero
        if all(c == n for c, n in zip(cancer_values, normal_values)):
            # If all values are identical, set the p-value to 1.0 (no difference)
            p_value = 1.0
        else:
            # Perform the Wilcoxon signed-rank test for the paired samples
            try:
                stat, p_value = wilcoxon(cancer_values, normal_values)
            except ValueError as e:
                print(f"Error with Wilcoxon test for gene {gene_name}: {e}. Skipping this gene.")
                continue

        # Append the results for this gene
        p_values.append({
            "gene_name": gene_name,
            "p_value": p_value
        })
    else:
        print(f"Warning: No common case_ids for gene {gene_name}. Skipping this gene.")

# Convert the results into a Polars DataFrame
p_values_df = pl.DataFrame(p_values)

# Apply Benjamini-Hochberg (BH) correction for multiple hypothesis testing
p_values_list = p_values_df["p_value"].to_list()  # Extract p-values as a list
_, corrected_p_values, _, _ = multitest.multipletests(p_values_list, method="fdr_bh")

# Add the corrected p-values to the DataFrame
p_values_df = p_values_df.with_columns(pl.Series("corrected_p_value", corrected_p_values))

# Display the p-values with BH correction
print("P-values with BH correction:")
print(p_values_df)

# Filter significant genes (e.g., corrected p-value < 0.05)
significant_genes_df = p_values_df.filter(pl.col("corrected_p_value") < 0.05)

# Display significant genes
print("Significant Genes (corrected p < 0.05):")
print(significant_genes_df)

#check if the random genes are in the significant_genes_df
random_genes_in_significant = random_genes.filter(
    pl.col("gene_name").is_in(significant_genes_df["gene_name"])
)

random_genes_in_significant['gene_name'].n_unique()

merged_significant_genes = significant_genes_df.join(
  merged_norm_cancer_df,
  on="gene_name",
  how="left")

merged_significant_genes = merged_significant_genes.with_columns(
  (merged_significant_genes["median_beta_value"] - merged_significant_genes["median_beta_value_normal"]).alias("effect_size")
)

merged_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")
)

# Now sort by the absolute effect size in descending order
merged_significant_genes = merged_significant_genes.sort(by="abs_effect_size", descending=True)

# Drop the temporary "abs_effect_size" column (if you don't need it)
merged_significant_genes = merged_significant_genes.drop("abs_effect_size","case_ids","case_ids_normal")

merged_significant_genes_pandas = merged_significant_genes.head().to_pandas()
merged_significant_genes_pandas.columns = [["gene_name", "p_value", "corrected_p_value",
        "median_beta_value_cancer", "median_beta_value_normal", "effect_size"]]

# top all differentially methylated genes
top_all_significant_genes = merged_significant_genes.with_columns(
    (pl.col("effect_size").abs()).alias("abs_effect_size")  # Take absolute value of effect_size
).sort("abs_effect_size", descending=True).head(1000)

#matching the top all differentially methylated genes to the methylation_cancer_df
methylation_top_all_df = methylation_cancer_df.filter(
    pl.col("gene_name").is_in(top_all_significant_genes["gene_name"])
)
