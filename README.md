# DS-Consulting-Capstone-Project
Capstone project for the "Data Science - Consulting Approach" class. The primary aim of this project was to conduct an in-depth analysis of customer satisfaction surveys to understand the customer service experience at an E-commerce platform Shopzilla. We developed an ETL process and a Power BI dashboard to examine both qualitative and quantitative aspects of customer feedback, identifying factors that impact customer satisfaction. The insights derived from this analysis aim to pinpoint areas for improvement and offer actionable recommendations to enhance the overall customer experience. The data comes from [Kaggle](https://www.kaggle.com/datasets/ddosad/ecommerce-customer-service-satisfaction).

To obtain the data used in the Power BI report, run scripts in the following order:
1. ```scripts/get_sentiment.py``` - obtain sentiment for each available customer remark
2. ```scripts/filter_agent_feedback.py``` - filter customer remarks related to the customer service
3. ```scripts/get_word_frequencies.py``` - obtain frequencies for each relevant word in the customer remarks
