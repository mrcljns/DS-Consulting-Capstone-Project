import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
tqdm.pandas()


def get_roberta_sentiment(text):
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative' : scores[0],
        'neutral' : scores[1],
        'positive' : scores[2]
    }

    return max(scores_dict, key=scores_dict.get)


if __name__ == '__main__':
    full_df = pd.read_csv('../data/Customer_support_data.csv')
    remarks_df = full_df[full_df['Customer Remarks'].isna() == False].reset_index(drop=True)
    MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    remarks_df['sentiment'] = remarks_df['Customer Remarks'].progress_map(lambda x: get_roberta_sentiment(str(x)))

    remarks_df.to_csv('../data/Customer_support_data_sentiment.csv', index=False)