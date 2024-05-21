import pandas as pd


def keyword_search(text):
    keyword_set = ('executive', 'staff', 'customer service', 'customer care', 'customer_service', 'agent', 'person', 'talk')
    
    return any(x in text.lower() for x in keyword_set)


if __name__ == '__main__':
    full_df = pd.read_csv('../data/Customer_support_data.csv')
    remarks_df = full_df[full_df['Customer Remarks'].isna() == False]
    remarks_df['keyword_in'] = remarks_df['Customer Remarks'].map(lambda x: keyword_search(str(x)))
    remarks_df = remarks_df[remarks_df.keyword_in == True]
    remarks_df = remarks_df.groupby('Agent_name').agg({'Customer Remarks': '; '.join}).reset_index()
    remarks_df.to_csv('../data/agent_feedback.csv', index=False)