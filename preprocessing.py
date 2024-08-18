import pandas as pd

# Load the CSV files
dsba_df = pd.read_csv('DSBA.csv')
ai_df = pd.read_csv('MScAI.csv')

# Define the mappings for DSBA and AI programmes
dsba_mapping = {
    "Big Data Analytics and Technologies": "BDAT",
    "Data Management": "DM",
    "Business Intelligence Systems": "BIS",
    "Multivariate Methods for Data Analysis": "MMDA",
    "Time Series Analysis and Forecasting": "DL",  # Merged
    "Behavioural Science, Social Media and Marketing Analytics": "BSSMA/CIS",  # Merged
    "Applied Machine Learning": "AML",
    "Advanced Business Analytics and Visualisation": "ABAV",
    "Data Analytical Programming": "DAP",
    "Research Methodology for Capstone Project": "RMCP",
    "Deep Learning": "DL",
    "Cloud Infrastructure and Services": "BSSMA/CIS",  # Merged
    "Operational Research and Optimisation": "ORO/BIA",  # Merged
    "Strategies in Emerging Markets": "SEM/DPM",  # Merged
    "Multilevel Data Analysis": "NLP",  # Merged
    "Data Protection and Management": "SEM/DPM",  # Merged
    "Building IoT Application": "ORO/BIA",  # Merged
    "Natural Language Processing": "NLP"
}

ai_mapping = {
    "Artificial Intelligence": "AI",
    "Image Processing and Computer Vision": "IPCV",
    "Business Intelligence Systems": "BIS",
    "Multivariate Methods for Data Analysis": "MMDA",
    "Fuzzy Logic": "FL",
    "Computational Intelligence Optimization": "CIO",
    "Applied Machine Learning": "AML",
    "Applied Robotics": "AR",
    "Pattern Recognition": "PR",
    "Research Methodology in Computing and Engineering": "RMCE",
    "Deep Learning": "DL",
    "Expert Systems and Knowledge Engineering": "ESKE",
    "Natural Language Processing": "NLP"
}

# List of all DSBA modules after merging
all_dsba_modules = ["BDAT", "DM", "BIS", "MMDA", "DL", "BSSMA/CIS", "AML", "ABAV", "DAP", "RMCP", "ORO/BIA", "SEM/DPM",
                    "NLP"]

# List of all AI modules
all_ai_modules = ["AI", "IPCV", "BIS", "MMDA", "FL", "CIO", "AML", "AR", "PR", "RMCE", "DL", "ESKE", "NLP"]


# Function to process the dataframes
def process_dataframe(df, mapping, programme_name, all_modules):
    # Keep only the necessary columns
    df = df.loc[:, ['NAME', 'MODULE NAME']]
    # Map the full module names to their short forms
    df.loc[:, 'MODULE NAME'] = df['MODULE NAME'].map(mapping)
    # Add the programme column
    df.loc[:, 'PROGRAMME'] = programme_name
    # Count the occurrences of each module
    count_df = df.groupby(['PROGRAMME', 'MODULE NAME']).size().reset_index(name='COUNT')
    # Ensure all modules are included, even with zero counts
    all_modules_df = pd.DataFrame({'MODULE NAME': all_modules, 'PROGRAMME': programme_name})
    result_df = all_modules_df.merge(count_df, on=['MODULE NAME', 'PROGRAMME'], how='left').fillna(0)
    return result_df


# Process both dataframes
dsba_processed = process_dataframe(dsba_df, dsba_mapping, 'DSBA', all_dsba_modules)
ai_processed = process_dataframe(ai_df, ai_mapping, 'AI', all_ai_modules)

# Combine the dataframes
final_result_df = pd.concat([dsba_processed, ai_processed])

# Save the result to a new CSV file
final_result_df.to_csv('processed_modules_combined.csv', index=False)
