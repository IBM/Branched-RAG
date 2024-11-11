import os
import nest_asyncio
import entity_extraction
#from data_collection import generate_data

nest_asyncio.apply()

from metrics import LLMEvaluator
# import openpyxl
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings

from genai import Client
from genai.credentials import Credentials
from genai.extensions.llama_index import IBMGenAILlamaIndex
from genai.schema import DecodingMethod, TextGenerationParameters

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.callbacks import CBEventType, EventPayload

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ibm import WatsonxLLM
from llama_index.core.query_engine import PGVectorSQLQueryEngine as QueryEngine
from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.vector_stores.milvus import MilvusVectorStore

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

def get_generated_text(context,query):
# make sure you have a .env file under genai root with
    GENAI_KEY='Enter GENAI_KEY here'
    load_dotenv()

    parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    }

    llm = WatsonxLLM(
    apikey="Enter WatsonX API Key here",
    model_id="ibm/granite-13b-instruct-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="Enter WatsonX Project ID here",
    params=parameters,
    )

    Settings.llm = llm

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    directory = 'MarketResearchDirectory'
    if not os.path.exists(directory):
        os.makedirs(directory)



    #Running the api's to generate data

    # entities = generate_data(query,directory)
    # print('Entities extracted', entities)

    data_dir = os.path.join(os.getcwd(),directory)
    data = SimpleDirectoryReader(input_dir=data_dir).load_data()
    # print("data",data)
###############

############### Integrating Milvus #################
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    # from ibm_watsonx_ai.foundation_models import Embeddings
    vector_store = MilvusVectorStore(
        uri="./milvus_demo.db", dim=10, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from ibm_watsonx_ai.foundation_models import Embeddings
    from ibm_watsonx_ai import Credentials
    from langchain.vectorstores import Milvus
    from pymilvus import (connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,list_collections
)
    host = "Enter Milvus Host here"
    port = "Enter Milvus Port here"

    password="Enter Passeword here"
    user = "Provide User here"

    connections.connect(
                    host = host,
                    port = port,
                    user = user,
                    password = password,
                    secure=True,
    server_name=host,
      )

    vector_db= MilvusVectorStore(dim=1536,
        embedding_function=Settings.embed_model,
        collection_name="milvus_schema",
        connection_args={"host": host, "port":port, "secure":True,
                                             "server_name": host ,"user":user,"password": password}

    )

    storage_context = StorageContext.from_defaults(vector_store=vector_db)
#     ################ Integrating Milvus #################

    # build index and query engine
    vector_query_engine = VectorStoreIndex.from_documents(
        data,
        storage_context=storage_context,
        # vector_store = vector_db,
        use_async=True,
    ).as_query_engine()



    # setup base query engine as tool
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="Data Analyser",
                description="Data Analyser - Analyse data from different data sources to compile the results from the financial query",
            ),
        ),
    ]


    # Retrieve context
    # context = retrieve_context(query)
    # print(f'context {context}')

    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
    )
    # Use the context in your query engine
    # response = query_engine.query(f"Context: {context}\n\nQuestion: {query}")


    response = query_engine.query(query)
    return response.response
# queries = ['In the airline industry, how have the stock performances of Delta, United, and Southwest been impacted by recent news regarding fuel prices, labor disputes, and post-pandemic travel demand?']
            # 'Which energy company among ExxonMobil, Chevron, and Shell has demonstrated the strongest financial metrics in terms of debt reduction, dividend sustainability, and free cash flow generation, and how has this affected their stock valuation?',]
            # "Give me some fundamentals of the following stocks Google, Nvidia, Meta, Netflix?",
            # 'Comparing Intel, AMD, and NVIDIA, which semiconductor company has seen the greatest stock price appreciation in the last 6 months, and how does this correlate with their reported revenue growth and market share gains?']
# queries = ["How to calculate HHI,What is price elasticity,what is Oligopoly theory,Give me the cost per Rs.100 earnings of indusind bank",
#            "Give me the cost per Rs.100 earnings of indusind bank"
#            "Give me the cost per Rs.100 earnings of HDFC bank"
#     "How have the stock prices of Amazon, Walmart, and Target fluctuated in response to their quarterly earnings reports over the past year, and which company has shown the most consistent beat on analyst expectations?",
#     "In the airline industry, how have the stock performances of Delta, United, and Southwest been impacted by recent news regarding fuel prices, labor disputes, and post-pandemic travel demand?",
#     "Give me some fundamentals of the following stocks Google, Nvidia, Meta, Netflix?",
#     "Comparing Intel, AMD, and NVIDIA, which semiconductor company has seen the greatest stock price appreciation in the last 6 months, and how does this correlate with their reported revenue growth and market share gains?",
#     "Compare the stock performance of Tesla, Ford, and General Motors in relation to their electric vehicle sales and market share growth over the last two quarters.",
#     "Which pharmaceutical companies among Pfizer, Moderna, and Johnson & Johnson have seen the most significant stock price movements following COVID-19 vaccine-related news?",
#     "How have the stock prices of major banks like JPMorgan Chase, Bank of America, and Wells Fargo been affected by recent changes in interest rates and economic indicators?",
#     "Compare the stock performance of streaming platforms Netflix, Disney+, and Amazon Prime Video in response to their subscriber growth and content investments over the past year.",
#     "Which renewable energy stocks among First Solar, Vestas Wind Systems, and NextEra Energy have shown the strongest performance in relation to government policy changes and technological advancements?",
#     "How have the stock prices of Coca-Cola, PepsiCo, and Dr Pepper Snapple Group responded to shifting consumer preferences towards healthier beverages?",
#     "Compare the stock performance of cybersecurity companies like Crowdstrike, Palo Alto Networks, and Fortinet in light of recent high-profile cyber attacks and increased corporate spending on security.",
#     "Which fast-food chains among McDonald's, Yum! Brands, and Chipotle Mexican Grill have seen the most significant stock price appreciation following their digital transformation efforts?",
#     "How have the stock prices of major hotel chains like Marriott, Hilton, and Hyatt responded to the recovery in global tourism and business travel?",
#     "Compare the stock performance of social media platforms Twitter, Snap, and Pinterest in relation to their user growth and advertising revenue over the past two quarters.",
#     "Which cloud computing stocks among Amazon Web Services, Microsoft Azure, and Google Cloud have shown the strongest correlation between their market share gains and stock price appreciation?",
#     "How have the stock prices of major retailers like Home Depot, Lowe's, and Costco been impacted by inflation and changes in consumer spending patterns?",
#     "Compare the stock performance of payment processors Visa, Mastercard, and PayPal in light of the growing adoption of digital payments and emerging fintech competitors.",
#     "Which gaming companies among Electronic Arts, Activision Blizzard, and Take-Two Interactive have seen the most significant stock price movements following major game releases or acquisition news?"
# ]

#####Evaluation code##############
# data=[]
# for query in queries:

#     response = get_generated_text('',query)

#     sub_answers=[]
#     for i, (start_event, end_event) in enumerate(
#         llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
#     ):
#         qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
#         # print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
#         # print("Answer: " + qa_pair.answer.strip())
#         # print("====================================")
#         sub_answers.append(qa_pair.answer.strip())
    # print('--------------------sub-answers-----------------------')
    # print(sub_answers)
    # print('-----------------------------EOS----------------------')
    # context = ' '.join(sub_answers)
    # context_relevance_score, context_relevance_reasons,answer_relevance_score, answer_relevance_reasons,faithfulness_score = evaluate(query,response,context)
    # data.append((query,response,context,context_relevance_score, context_relevance_reasons,answer_relevance_score, answer_relevance_reasons,faithfulness_score))

# df = pd.DataFrame(data, columns=['Query', 'Response', 'Context','context_relevance_score','context_relevance_reasons','answer_relevance_score','answer_relevance_reasons','faithfulness_score'])
# df.to_csv('results_branched_rag_BIAN.csv',index = False)
# print(df.head())
##############################################################
    # for i, (start_event, end_event) in enumerate(
    #     llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
    # ):
    #     qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    #     print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    #     print("Answer: " + qa_pair.answer.strip())
    #     print("====================================")

    # d = {}
    # for question in questions:
    #     sub_question =[]
    #     response = query_engine.query(
    #     str(question))
    #     # iterate through sub_question items captured in SUB_QUESTION event
    #     for i, (start_event, end_event) in enumerate(
    #         llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
    #     ):
    #         qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    #         print(qa_pair.sub_q.sub_question.strip())
    #         sub_question.append(qa_pair.sub_q.sub_question.strip())
    #         # print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    #         # print("Answer: " + qa_pair.answer.strip())
    #         print("====================================")
    #     d[question]=sub_question
    #     print(response)

    # sub_ques_json = json.dumps(d,indent=4)
    # with open('sub_questions.json', 'w') as file:
    #     file.write(sub_ques_json)
    # # print(d)
    # print(response)

# questions = [
#         "How have tech giants Apple, Google, and Microsoft performed in the stock market over the past year, and which one has shown the most consistent growth in terms of quarterly earnings?",
#         "What are the top 3 performing sectors in the S&P 500 this quarter, and how do their performances compare to the same period last year?",
        # "Can you provide an analysis of the correlation between oil prices and renewable energy stocks over the last five years? Which renewable energy companies have shown the most resilience during oil price fluctuations?",
        # "How has the Federal Reserve's interest rate policy affected the banking sector stocks in the past two years, and what are analysts predicting for regional banks versus large national banks in the coming year?",
        # "What impact did the COVID-19 pandemic have on the airline industry stocks, and how do their current valuations compare to pre-pandemic levels? Which airline has shown the strongest recovery?",
        # "How do ESG (Environmental, Social, and Governance) scores correlate with stock performance in the technology sector? Can you provide examples of high-performing tech stocks with both good and poor ESG ratings?",
        # "What are the top-performing cryptocurrency-related stocks in the past year, and how do their performances compare to major cryptocurrencies like Bitcoin and Ethereum?",
        # "How have geopolitical tensions between the US and China affected semiconductor stocks? Which companies in this sector have shown the most growth despite these challenges?",
        # "Can you compare the performance of traditional automotive stocks versus electric vehicle stocks over the past three years? How have government policies in different countries influenced this trend?",
        # "What is the relationship between inflation rates and gold mining stocks over the last decade? How does this compare to the performance of gold ETFs during the same period?",
        # "How have supply chain disruptions affected retail stocks in the past two years, and which companies have successfully adapted their strategies to mitigate these issues?",
        # "What are the top-performing artificial intelligence and machine learning focused stocks, and how do their financials compare to more established tech giants?",
        # "How have healthcare stocks performed during major global health crises in the past 20 years, including the COVID-19 pandemic? Which subsectors (e.g., pharmaceuticals, medical devices, telehealth) showed the most growth?",
        # "Can you analyze the impact of streaming services on traditional media and entertainment stocks? How have companies like Disney, which have both traditional and streaming offerings, fared in comparison to pure-play streaming services like Netflix?",
        # "What is the correlation between climate-related natural disasters and insurance company stocks? How have insurers adapted their strategies, and which companies have shown the most resilience?",
        # "How have changes in remote work trends affected commercial real estate stocks versus residential real estate stocks? Can you provide examples of companies that have successfully pivoted their strategies in response to these changes?",
        # "What is the relationship between agricultural commodity prices and farming equipment manufacturer stocks? How have recent global events (e.g., conflicts, climate change) influenced this relationship?",
        # "Can you compare the stock performance of major social media companies over the past five years? How have issues like data privacy concerns, regulatory challenges, and shifting user demographics affected their valuations?",
        # "How have space exploration and satellite communication stocks performed in recent years? What are the key factors driving growth in this sector, and which companies are leading in terms of innovation and market share?",
        # "What is the impact of 5G technology rollout on telecommunication stocks versus semiconductor stocks? How have regulatory decisions in different countries affected the performance of companies in these sectors?"
# ]
# questions = ['What is the stock price of apple today?']
# for question in questions:
#     response = get_generated_text('',question)
# print(response)


