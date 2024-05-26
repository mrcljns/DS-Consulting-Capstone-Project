import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
tqdm.pandas()


# Define a function to get sentiment using RoBERTa model
def get_roberta_sentiment(text):
    # Tokenize the input text
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    
    # Get the model's output
    output = model(**encoded_text)
    
    # Extract and process the scores
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Create a dictionary for sentiment scores
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }

    # Return the sentiment with the highest score
    return max(scores_dict, key=scores_dict.get)

if __name__ == '__main__':
    # Load the customer support data from a CSV file into a DataFrame
    full_df = pd.read_csv('../data/Customer_support_data.csv')
    
    # Filter out rows where 'Customer Remarks' column is not empty and reset the index
    remarks_df = full_df[full_df['Customer Remarks'].isna() == False].reset_index(drop=True)
    
    # Define the RoBERTa model and tokenizer
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    # Apply get_roberta_sentiment function to each customer remark and store the result in a new column 'sentiment'
    tqdm.pandas()  # Enable progress bar for the map function
    remarks_df['sentiment'] = remarks_df['Customer Remarks'].progress_map(lambda x: get_roberta_sentiment(str(x)))
    
    # Merge the sentiment results back into the original DataFrame
    full_df = pd.merge(full_df, remarks_df[['Unique id', 'sentiment']], on='Unique id', how='left')
    
    # Fill any missing sentiment values with 'neutral'
    full_df['sentiment'] = full_df['sentiment'].fillna('neutral')
    
    # Save the resulting DataFrame to a new CSV file
    full_df.to_csv('../data/Customer_support_data_sentiment.csv', index=False)