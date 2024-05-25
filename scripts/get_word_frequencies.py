import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


# Function to remove digits from text
def remove_digits(text):
    return re.sub(r'\d+', '', text)


# Function to remove specific keywords from text
def remove_keywords(text, keywords):
    # Join keywords with '|' to create a regex pattern
    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
    # Remove keywords using regex
    return re.sub(pattern, '', text, flags=re.IGNORECASE)


if __name__ == '__main__':
    keyword_set = ('executive', 'staff', 'customer', 'service', 'care', 'consultant', 
                   'agent', 'person', 'talk', 'seller', 'shopzilla', 'product')
        
    # Setting a custom count vectorizer to remove stopwords
    vectorizer_model = CountVectorizer(stop_words="english")

    # Read the feedback data from the CSV file
    feedback_df = pd.read_csv('../data/agent_feedback.csv')

    # Remove digits from 'Customer Remarks' column
    tqdm.pandas()  # Enable progress bar for apply function
    feedback_df['Customer Remarks'] = feedback_df['Customer Remarks'].progress_apply(lambda x: remove_keywords(remove_digits(x), keyword_set))

    # Group comments by agent and combine them into a single string
    feedback_df = feedback_df.groupby('Agent_name')['Customer Remarks'].apply(' '.join).reset_index()

    # Fit and transform the combined comments to get the word counts
    frequencies = vectorizer_model.fit_transform(feedback_df['Customer Remarks'])

    # Create a DataFrame with the word counts
    word_freq_df = pd.DataFrame(frequencies.toarray(), columns=vectorizer_model.get_feature_names_out(), index=feedback_df['Agent_name'])

    # Reset the index to make 'Agent_name' a column
    word_freq_df = word_freq_df.reset_index()

    # Unpivot the DataFrame from wide to long format
    word_freq_long_df = pd.melt(word_freq_df, id_vars=['Agent_name'], var_name='word', value_name='frequency')

    # Filter out rows with zero frequency to reduce size
    word_freq_long_df = word_freq_long_df[word_freq_long_df['frequency'] > 0]

    # Sort the DataFrame by frequency in descending order
    word_freq_long_df.sort_values('frequency', ascending=False, inplace=True)

    # Save the resulting DataFrame to a CSV file
    word_freq_long_df.to_csv('../data/user_word_frequencies_long.csv', index=False)