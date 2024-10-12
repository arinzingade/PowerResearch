from googlesearch import search
import yfinance as yf
import os
from langchain.prompts import ChatPromptTemplate
from firecrawl import FirecrawlApp
import asyncio
import getpass
import os
from datetime import datetime
from hashlib import md5
from typing import Dict, List
from groq import Groq
import pandas as pd
import tiktoken
from langchain_community.graphs import Neo4jGraph
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from datetime import datetime
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv()

import yfinance as yf

class Finance:
    def get_finance_data(self, ticker):

        stock_ticker = yf.Ticker(ticker)
        df_balancesheet = stock_ticker.balance_sheet
        balance_sheet_json = df_balancesheet.to_json()
        df_incomestmt = stock_ticker.financials
        incomestmt_json = df_incomestmt.to_json()
        df_cashflow = stock_ticker.cashflow
        cashflow_json = df_cashflow.to_json()

        return balance_sheet_json, incomestmt_json, cashflow_json

class WebScrape:
    def __init__(self):
        self.app = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API'))
        self.useful_text = []
    
    def get_top_links(self, stock_name, query, num_results=1):
        top_links = []
        search_results = search(query, num_results=num_results)
        
        for result in search_results:
            top_links.append(result)
        
        return top_links

    def top_links(self, stock_name):
        search_news = stock_name + " latest stock news"
        search_latest_opinion = stock_name + " latest financial analysis"
        global_financial_news = "Latest News Financial Markets International"
        latest_filings = stock_name + " Latest Legal Filings / Reporting"

        top_news_links = self.get_top_links(stock_name, search_news)
        top_blogs_links = self.get_top_links(stock_name, search_latest_opinion)
        top_global_links = self.get_top_links(stock_name, global_financial_news)
        top_filings_links = self.get_top_links(stock_name, latest_filings)
        top_news_links.append('https://tradingeconomics.com/stream')

        all_links_combined = (
            top_news_links + 
            top_blogs_links + 
            top_global_links + 
            top_filings_links
        )

        return all_links_combined

    def chunk_text(text, size=50000):
        return [text[i:i + size] for i in range(0, len(text), size)]
    


    def return_useful_text(self, all_links_combined):

        PROMPT_TEMPLATE = """
        Prompt: Extract only the useful information from the below text. 
        Do not include links. Always include plain text full of useful, 
        high-impact information in a concise manner.
        {context}
        """

        for link in all_links_combined:
            try:
                scrape_result = self.app.scrape_url(link, params={'formats': ['markdown', 'html']})
                large_text = scrape_result['markdown']
                chunked_text = self.chunk_text(large_text)

                for chunk in chunked_text:
                    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    prompt = prompt_template.format(context=chunk)
                    response_text = Groq.call_groq(prompt)

                    self.useful_text.append(response_text)
                    print(response_text)
            
            except requests.exceptions.HTTPError as e:
                print(f"Request failed for {link}: {e}. Skipping...")
                continue  
            except Exception as e:
                print(f"An unexpected error occurred for {link}: {e}. Skipping...")
                continue  

        whole_text = "".join(self.useful_text)
        return whole_text
        
class Groq:

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY'),
        )



    def call_groq(self, scraped_text_chunk):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": scraped_text_chunk,
                }
            ],
            model="llama3-8b-8192", 
        )

        return chat_completion.choices[0].message.content
    

class Neo4j:

    def __init__(self):
        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')

        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        self.init_graph()

        self.construction_chain = self.data_llm_model()

    def init_graph(self):
        graph = Neo4jGraph(refresh_schema=False)
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:AtomicFact) REQUIRE c.id IS UNIQUE")
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:KeyElement) REQUIRE c.id IS UNIQUE")
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

        return graph


    construction_system = """
    You are an intelligent assistant tasked with meticulously extracting key elements and atomic facts from the text.
    1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
    2. Extract key elements and atomic facts comprehensively, without omitting query-worthy details.
    3. Replace pronouns with specific nouns where applicable.
    4. Ensure the extracted key elements and atomic facts match the language of the original text.
    """

    construction_human = "Use the following format to extract information from the input: {input}"

    construction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                construction_system,
            ),
            (
                "human",
                (
                    "Use the given format to extract information from the "
                    "following input: {input}"
                ),
            ),
        ]
    )



    def data_llm_model(self):
        model = ChatGroq(model="gemma2-9b-it", temperature=0.1)
        structured_llm = model.with_structured_output(self.Extraction)
        construction_chain = self.construction_prompt | structured_llm
        return construction_chain



    def encode_md5(self, text):
        return md5(text.encode("utf-8")).hexdigest()
    

    # Paper used 2k token size
    async def process_document(text, document_name, graph, chunk_size=2000, chunk_overlap=200):
        start = datetime.now()
        print(f"Started extraction at: {start}")
        
        # Split text into chunks
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_text(text)
        print(f"Total text chunks: {len(texts)}")

        # Create async tasks for LLM extraction for each chunk
        tasks = [
            asyncio.create_task(self.construction_chain.ainvoke({"input": chunk_text}))
            for index, chunk_text in enumerate(texts)
        ]
        
        # Await all tasks to complete
        results = await asyncio.gather(*tasks)
        print(f"Finished LLM extraction after: {datetime.now() - start}")

        # Process results into the document structure
        docs = []
        for index, result in enumerate(results):
            doc = result.dict()
            doc['chunk_id'] = self.encode_md5(texts[index])
            doc['chunk_text'] = texts[index]
            doc['index'] = index
            for af in doc["atomic_facts"]:
                af["id"] = encode_md5(af["atomic_fact"])
            docs.append(doc)
            import_query = """
            MERGE (d:Document {id:$document_name})
            WITH d
            UNWIND $data AS row
            MERGE (c:Chunk {id: row.chunk_id})
            SET c.text = row.chunk_text,
                c.index = row.index,
                c.document_name = row.document_name
            MERGE (d)-[:HAS_CHUNK]->(c)
            WITH c, row
            UNWIND row.atomic_facts AS af
            MERGE (a:AtomicFact {id: af.id})
            SET a.text = af.atomic_fact
            MERGE (c)-[:HAS_ATOMIC_FACT]->(a)
            WITH c, a, af
            UNWIND af.key_elements AS ke
            MERGE (k:KeyElement {id: ke})
            MERGE (a)-[:HAS_KEY_ELEMENT]->(k)
            """

        graph.query(import_query, params={"data": docs, "document_name": document_name})

        # Create NEXT relationships between chunks
        graph.query(
            """
            MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            WHERE d.id = $document_name
            WITH c ORDER BY c.index 
            WITH collect(c) AS nodes
            UNWIND range(0, size(nodes) - 2) AS index
            WITH nodes[index] AS start, nodes[index + 1] AS end
            MERGE (start)-[:NEXT]->(end)
            """, 
            params={"document_name": document_name}
        )
        
        print(f"Finished import at: {datetime.now() - start}")
    

class AtomicFact(BaseModel):
    key_elements: List[str] = Field(description="""The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g.,
    actions), and adjectives (e.g., states, feelings) that are pivotal to the atomic fact's narrative.""")
    atomic_fact: str = Field(description="""The smallest, indivisible facts, presented as concise sentences. These include
    propositions, theories, existences, concepts, and implicit elements like logic, causality, event
    sequences, interpersonal relationships, timelines, etc.""")

class Extraction(BaseModel):
    atomic_facts: List[AtomicFact] = Field(description="List of atomic facts")