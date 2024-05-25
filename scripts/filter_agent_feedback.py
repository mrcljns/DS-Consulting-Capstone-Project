import pandas as pd


import pandas as pd

# Define a function to search for keywords in a given text
def keyword_search(text, keyword_set):
    # Check if any keyword in the keyword_set is present in the text (case insensitive)
    return any(x in text.lower() for x in keyword_set)

if __name__ == '__main__':
    # Define a set of keywords to search for in the customer remarks
    keyword_set = ('executive', 'staff', 'customer service', 'customer care', 
                   'customer_service', 'agent', 'person', 'talk', 'seller')
    
    # Load the customer support data from a CSV file into a DataFrame
    full_df = pd.read_csv('../data/Customer_support_data.csv')
    
    # Filter out rows where 'Customer Remarks' column is not empty
    remarks_df = full_df[full_df['Customer Remarks'].isna() == False]
    
    # Apply keyword_search function to each customer remark and store the result in a new column 'keyword_in'
    remarks_df['keyword_in'] = remarks_df['Customer Remarks'].map(lambda x: keyword_search(str(x), keyword_set))
    
    # Filter the DataFrame to include only rows where the 'keyword_in' column is True
    remarks_df = remarks_df[remarks_df.keyword_in == True]
    
    # Group the DataFrame by 'Agent_name' and aggregate customer remarks by joining them with '; '
    remarks_df = remarks_df.groupby('Agent_name').agg({'Customer Remarks': '; '.join}).reset_index()
    
    # Save the resulting DataFrame to a new CSV file
    remarks_df.to_csv('../data/agent_feedback.csv', index=False)
