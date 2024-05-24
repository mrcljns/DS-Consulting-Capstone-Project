import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, LlamaCPP
from llama_cpp import Llama
from sklearn.feature_extraction.text import CountVectorizer


def keyword_search(text):
    keyword_set = ('executive', 'staff', 'customer service', 'customer care', 'customer_service', 'agent', 'person', 'talk', 'seller')
    
    return any(x in text.lower() for x in keyword_set)

    
def train_bert(docs, llm_path, prompt, save_path):
    # Setting a custom count vectorizer will help in removing stopwords
    vectorizer_model = CountVectorizer(stop_words="english")
    llm = Llama(model_path=llm_path, n_gpu_layers=-1, n_ctx=2048, max_tokens=512, stop=["Q:", "\n"])
    # Create a pipeline for the bert model
    representation_model = {
        'KeyBERT': KeyBERTInspired(),
        'LLM': LlamaCPP(llm, prompt=prompt)
        }
    topic_model = BERTopic(representation_model=representation_model, 
                           top_n_words=10, 
                           n_gram_range=(1, 2), 
                           vectorizer_model=vectorizer_model, 
                           verbose=True)
    # Fit the model on customer remarks and save it to a designated path
    topics, probs = topic_model.fit_transform(docs)
    topic_model.save(save_path, serialization='safetensors', save_ctfidf=True, save_embedding_model='all-MiniLM-L6-v2')
    return topic_model


if __name__ == '__main__':
    full_df = pd.read_csv('../data/Customer_support_data_sentiment.csv')
    remarks_df = full_df[full_df['sentiment'] != 'neutral']
    remarks_df['keyword_in'] = remarks_df['Customer Remarks'].map(lambda x: keyword_search(str(x)))
    remarks_df = remarks_df[remarks_df.keyword_in == True].reset_index(drop=True)
    prompt = """
    Q: I have a topic that contains the following customer remarks about the service: 
    [DOCUMENTS]

    The topic is described by the following keywords: '[KEYWORDS]'.

    Based on the above information, can you give a short label of the topic?
    A: 
    """
    topic_model = train_bert(docs=list(remarks_df['Customer Remarks']), 
                             llm_path='../models/zephyr-7b-alpha.Q4_K_M.gguf', 
                             prompt=prompt, 
                             save_path='../models/ecommerce_topic')
    # Save the list of topics into a csv
    topic_model.get_topic_info().to_csv('../data/topic_info.csv', index=False)
    # Save the most frequent topics for each agent
    agent_topics = topic_model.topics_per_class(docs=remarks_df['Customer Remarks'], classes=remarks_df['Agent_name']).to_csv('../data/agent_topics.csv', index=False)