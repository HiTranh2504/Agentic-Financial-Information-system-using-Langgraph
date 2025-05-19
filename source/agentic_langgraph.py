import os
import re
import json
import yaml
import numpy as np
import pandas as pd
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from finance_metrics import load_formulas, identify_metric, get_required_fields, compute_metric
from plot_metric import plot_metric
from langgraph.graph import StateGraph, END
from IPython.display import Markdown, display


# =======================Setup Enviroment Variables=======================
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# =======================Build Vector Database=======================
chunks = None
index = None
embedding_model = None

def get_vector_store_and_retriever(resource_dir: str = "sec_embeddings") -> Tuple[List[Dict[str, Any]], faiss.Index, SentenceTransformer]:
    global chunks, index, embedding_model
    
    if chunks is not None and index is not None and embedding_model is not None:
        print("✅ Vector store & retriever already initialized. Reusing...")
        return chunks, index, embedding_model

    try:
        print(f"Loading RAG vector store from: {resource_dir}")
        
        with open(os.path.join(resource_dir, "chunks.json"), 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        with open(os.path.join(resource_dir, "embeddings.pkl"), 'rb') as f:
            embeddings = pickle.load(f)
        
        index = faiss.read_index(os.path.join(resource_dir, "faiss_index.bin"))
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"✅ Loaded {len(chunks)} chunks into memory.")
        print("✅ FAISS index and SentenceTransformer initialized.")

    except Exception as e:
        print(f"❌ Failed to load vector store and retriever: {e}")
        chunks = None
        index = None
        embedding_model = None

    return chunks, index, embedding_model


# =======================Create A Query Retrieval Grader=======================
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Parser
parser = StrOutputParser()

# Prompt system
SYS_PROMPT = """
You are an expert grader assessing relevance of a retrieved document to a user question.
Answer only 'yes' or 'no' depending on whether the document is relevant to the question.
"""

# create prompt template
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    ("human", "Retrieved document:\n{document}\n\nUser question:\n{question}")
])

# create chain
doc_grader = grade_prompt | llm | parser

# =======================Build A QA RAG Chain=======================
# Create RAG prompt for response generation
prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If no context is present or if you don't know the answer, just say that you don't know the answer.
Do not make up the answer unless it is there in the provided context.
Give a detailed answer and to the point answer with regard to the question.
Question:
{question}
Context:
{context}
Answer:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)

# Initialize connection with gpt-4.1-mini
chatgpt = ChatOpenAI(model_name='gpt-4.1-mini', temperature=0)

# Used for separating context docs with new lines
def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

# Create QA RAG chain
qa_rag_chain = (
	{
		"context": (itemgetter('context')
					|
					RunnableLambda(format_docs)),
		"question": itemgetter('question')
	}
	|
	prompt_template
	|
	chatgpt
	|
	StrOutputParser()
)

# =======================Create A Query Rephraser=======================
# LLM for question rewriting
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# Prompt template for rewriting
SYS_PROMPT = """Act as a question re-writer and perform the following task:
				 - Convert the following input question to a better version that is optimized for web search.
				 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
			 """
re_write_prompt = ChatPromptTemplate.from_messages(
	[
		("system", SYS_PROMPT),
		("human", """Here is the initial question:
					 {question}
					 Formulate an improved question.
				  """)
	]
)

# Create rephraser chain
question_rewriter = (re_write_prompt|llm|StrOutputParser())


# =======================Load Web Search Tool=======================
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)


# =======================SQL Query Generator=======================
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DBNAME") or os.getenv("POSTGRES_DB"),
    "user": os.getenv("DBUSER") or os.getenv("POSTGRES_USER"),
    "password": os.getenv("DBPASSWORD") or os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("DBHOST") or "localhost",
    "port": int(os.getenv("DBPORT", 5432)),
    "sslmode": os.getenv("SSL_MODE", "require")
}


print(f"DB Config: {DB_CONFIG}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def connect_to_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Kết nối thành công đến PostgreSQL.")
        return conn
    except Exception as e:
        print(f"Lỗi kết nối cơ sở dữ liệu: {e}")
        return None

def get_schema_and_samples(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        tables = cursor.fetchall()
        schema_info = {}
        for (table,) in tables:
            cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
            schema_info[table] = [{"column_name": col, "data_type": dtype} for col, dtype in cursor.fetchall()]
            cursor.execute(f'SELECT * FROM "{table}" LIMIT 3;')
            sample_rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            schema_info[f"{table}_samples"] = [dict(zip(colnames, row)) for row in sample_rows]
        cursor.close()
        return schema_info
    except Exception as e:
        return {"error": str(e)}

def generate_sql_query(user_question, schema_info=None):
    prompt = f"""
You are a PostgreSQL expert. You are working with a financial database containing two structured tables: "djia_prices" and "djia_companies".

IMPORTANT:
PostgreSQL is case-sensitive **only when using quoted identifiers**.
You MUST wrap column names in double quotes (e.g. "Ticker", "Close", "Date", etc.)
Do NOT use unquoted identifiers like Ticker or Close - they will cause errors.

Table 1: "djia_prices" - Daily stock trading data per company
- "Date" (TIMESTAMPTZ): The timestamp of the trading day.
- "Open" (FLOAT): Opening price of the stock on that day.
- "High" (FLOAT): Highest price reached during the trading day.
- "Low" (FLOAT): Lowest price of the stock during the trading day.
- "Close" (FLOAT): Closing price of the stock on that day.
- "Volume" (BIGINT): Number of shares traded on that day.
- "Dividends" (FLOAT): Dividend payout on that day, if any.
- "Stock_Splits" (FLOAT): Stock split ratio applied on that day (e.g. 2.0 = 2-for-1 split).
- "Ticker" (VARCHAR): The stock symbol of the company (e.g., AAPL, MSFT).

Table 2: "djia_companies" - Company profile information
- "symbol" (VARCHAR): The stock symbol matching the "Ticker" in "djia_prices".
- "name" (TEXT): Full name of the company.
- "sector" (TEXT): Main sector of operation (e.g., Technology, Healthcare).
- "industry" (TEXT): More specific industry classification (e.g., Consumer Electronics).
- "country" (TEXT): Country where the company is headquartered.
- "market_cap" (FLOAT): Market capitalization in USD.
- "pe_ratio" (FLOAT): Price-to-Earnings ratio of the company.
- "dividend_yield" (FLOAT): Annual dividend yield expressed as a percentage.
- "description" (TEXT): A short textual description of the company's operations.

Use this schema to write the most accurate and optimized PostgreSQL query to answer the following question.

Do NOT format the query in markdown or explain the result.
Return ONLY the raw SQL.

User Question:
{user_question}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        sql_query = response.choices[0].message.content
        sql_query = re.sub(r'^```sql', '', sql_query).strip('` \n')
        return sql_query
    except Exception as e:
        print(f"❌ Lỗi sinh SQL: {e}")
        return None


def execute_sql_query(conn, query):
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ Lỗi thực thi SQL: {e}")
        return None

def run_chat(question: str):
    conn = connect_to_database()
    if conn is None:
        return

    schema_info = get_schema_and_samples(conn)
    sql_query = generate_sql_query(question, schema_info)

    if not sql_query:
        print("❌ Không thể sinh truy vấn SQL.")
        return

    print(f"\n🧠 SQL sinh ra:\n{sql_query}")
    results = execute_sql_query(conn, sql_query)

    if results is None:
        print("❌ Truy vấn lỗi.")
        return

    print("\n📊 Kết quả:")
    print(results.head(5))

    conn.close()


# =======================Build Agentic RAG components=======================

# Graph State
from typing import List
from typing_extensions import TypedDict
class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]
documents: List[str]

# Retrieve function for retrieval from Vector DB
def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    documents = []

    chunks, index, model = get_vector_store_and_retriever(resource_dir="sec_embeddings")

    if index and chunks and model:
        try:
            # Encode query
            query_embedding = model.encode([question])[0].reshape(1, -1).astype(np.float32)

            # Search in the index
            distances, indices = index.search(query_embedding, 3)
            print(f"📌 Retrieved top {len(indices[0])} docs from FAISS")

            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    content = chunk["content"]
                    metadata = chunk.get("metadata", {})
                    score = float(1.0 / (1.0 + distances[0][i]))
                    doc = Document(page_content=content, metadata=metadata)
                    doc.metadata["score"] = score
                    documents.append(doc)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
    else:
        print("❌ No index/model/chunks available.")

    return {"documents": documents, "question": question}

# Grade documents
def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []

    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.strip().lower()

            if grade == "yes":
                print("✅ GRADE: DOCUMENT RELEVANT")
                filtered_docs.append(d)
            else:
                print("❌ GRADE: DOCUMENT NOT RELEVANT")
    else:
        print("Không có tài liệu nào từ vector DB.")

    return {
        **state,
        "documents": filtered_docs
    }

# Use SQL
# Danh sách 30 mã cổ phiếu DJIA (bạn có thể cập nhật thêm nếu cần)
DJIA_TICKERS = {
    "AAPL", "MSFT", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT"
}

def extract_ticker_from_question(question: str) -> str | None:
    """
    Extracts stock ticker symbol from user question by matching against DJIA tickers.
    Returns the first match found, or None if no match.
    """
    question = question.upper()
    for ticker in DJIA_TICKERS:
        if re.search(rf"\\b{ticker}\\b", question):
            print(f"✅ Detected ticker in question: {ticker}")
            return ticker
    print("⚠️ No DJIA ticker found in question.")
    return None


def detect_chart_type(question: str):
    q = question.lower()
    if "scatter plot" in q:
        return "scatter"
    if "bar chart" in q:
        return "bar"
    if "pie chart" in q:
        return "pie"
    if "boxplot" in q:
        return "box"
    if "histogram" in q:
        return "hist"
    if "heatmap" in q:
        return "heatmap"
    return None

def query_sql(state):
    print("---EXECUTE RAW SQL QUERY OR METRIC COMPUTATION---")
    question = state["question"]
    metadata = load_formulas()
    category, metric_name = identify_metric(question, metadata)

    conn = connect_to_database()
    if conn is None:
        raise ValueError("Không thể kết nối cơ sở dữ liệu.")

    if metric_name:
        print(f"📌 Identified metric: {metric_name} ({category})")

        if category == "TA":
            ticker = extract_ticker_from_question(question)
            if not ticker:
                conn.close()
                raise ValueError("❌ Cannot determine stock ticker from the question.")

            query = (
                f'SELECT "Date", "Open", "High", "Low", "Close", "Volume" '
                f'FROM djia_prices WHERE "Ticker" = \'{ticker}\' ORDER BY "Date" ASC'
            )
            print(f"🧠 SQL Query:\n{query}")
            df = pd.read_sql(query, conn)
            conn.close()
            df.rename(columns={"Date": "date"}, inplace=True)


            image_base64 = plot_metric(metric_name, df)
            result_doc = Document(
                page_content=f"Chart of {metric_name} for AAPL:",
                metadata={"image_base64": image_base64}
            )
            return {
                "documents": state["documents"] + [result_doc],
                "question": question
            }

        elif category == "FA":
            ticker = extract_ticker_from_question(question)
            if not ticker:
                conn.close()
                raise ValueError("❌ Cannot determine stock ticker from the question.")

            required_fields = get_required_fields(category, metric_name, metadata)
            query = f"SELECT {', '.join(required_fields)} FROM djia_companies WHERE symbol = '{ticker}' LIMIT 1;"
            print(f"🧠 SQL Query:\n{query}")
            df = pd.read_sql(query, conn)
            conn.close()


            if df.empty:
                result_str = f"⚠️ Không tìm thấy đủ dữ liệu để tính {metric_name}."
            else:
                result = compute_metric(metric_name, df.iloc[0].to_dict())
                result_str = f"{metric_name} = {result}"

            return {
                "documents": state["documents"] + [Document(page_content=result_str)],
                "question": question
            }

    # Natural language → SQL case
    schema_info = get_schema_and_samples(conn)
    sql_query = generate_sql_query(question, schema_info)
    if not sql_query:
        conn.close()
        raise ValueError("❌ Cannot generate SQL query from question.")

    print(f"🧠 Generated SQL Query:\n{sql_query}")
    df = execute_sql_query(conn, sql_query)
    conn.close()

    if df is None or df.empty:
        content = "⚠️ No results returned from SQL query."
        state["web_search_needed"] = "Yes"
        return {
            "documents": state["documents"],
            "question": question,
            "sql_query": sql_query,
            "web_search_needed": "Yes"
        }

    chart_type = detect_chart_type(question)
    if chart_type:
        print(f"📊 Detected chart type: {chart_type}")
        image_base64 = plot_fa_chart(df, chart_type, question)
        result_doc = Document(
            page_content=f"Visualization for: {question}",
            metadata={"image_base64": image_base64}
        )
    else:
        result_doc = Document(
            page_content=f"📊 SQL Query Result:\n\n{df.to_markdown(index=False)}"
        )

    return {
        "documents": state["documents"] + [result_doc],
        "question": question,
        "sql_query": sql_query,
        "web_search_needed": "No"
    }




# 
def assess_combined_documents(state):
    """
    Re-assess all documents (from vector + SQL) to determine which are relevant to the question.
    Filters out irrelevant documents based on LLM scoring.
    """
    print("---REASSESS COMBINED DOCUMENTS---")

    question = state.get("question", "")
    all_docs = state.get("documents", [])

    reassessed_docs = []

    for doc in all_docs:
        score = doc_grader.invoke({"question": question, "document": doc.page_content}).strip().lower()
        if score == "yes":
            print("✅ Reassessed: Relevant")
            reassessed_docs.append(doc)
        else:
            print("❌ Reassessed: Not relevant")

    print(f"🔍 Number of relevant documents after reassessment: {len(reassessed_docs)}")

    return {
        **state,
        "documents": reassessed_docs
    }


def decide_after_reassessment(state):
    """
    After reassessing documents from both vector DB and SQL,
    decide whether to proceed to generate an answer or fall back to web search.
    """
    docs = state.get("documents", [])
    num_docs = len(docs)
    print(f"🔍 Number of relevant documents after reassessment: {num_docs}")

    if num_docs >= 1:
        print("✅ Enough relevant documents found → proceed to generate_answer")
        return "generate_answer"
    else:
        print("⚠️ Not enough relevant documents → fallback to rewrite_query")
        return "rewrite_query"


def rewrite_query(state):
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_after_sql(state):
    if state.get("web_search_needed", "No") == "Yes":
        print("---DECISION: Missing SQL data → web search---")
        return "rewrite_query"
    
    print("---DECISION: Sufficient SQL data → generate answer---")
    return "generate_answer"




from langchain_core.documents import Document  # dùng đúng version core

def web_search(state):
    """
    Perform web search based on the rewritten question and return a unified Document.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    try:
        docs = tv_search.invoke(question)

        if not docs:
            print("⚠️ No results returned from web search.")
            return {"documents": documents, "question": question}

        print("📥 Web search results received.")
        print("DOCS TYPE:", type(docs))
        print("FIRST ELEMENT TYPE:", type(docs[0]) if docs else "EMPTY")

        # Format content
        if isinstance(docs[0], str):
            web_content = "\n\n".join(docs)
        elif isinstance(docs[0], dict) and "content" in docs[0]:
            web_content = "\n\n".join([d["content"] for d in docs])
        elif isinstance(docs[0], Document):
            web_content = "\n\n".join([d.page_content for d in docs])
        else:
            raise TypeError(f"Unsupported document format: {type(docs[0])}")

        # Append web search result as 1 Document
        web_results = Document(
            page_content=web_content,
            metadata={"source": "web_search"}
        )

        documents.append(web_results)
        print("✅ Web search content appended to documents.")
        return {"documents": documents, "question": question}

    except Exception as e:
        print(f"❌ Error during web search: {str(e)}")
        return {"documents": documents, "question": question}


def generate_answer(state):
    """
    Generate final answer using question and provided context documents.
    """
    print("---GENERATE ANSWER---")

    question = state.get("question", "")
    documents = state.get("documents", [])

    if not question:
        print("⚠️ No question provided.")
        return {**state, "generation": "No question was given."}

    if not documents:
        print("⚠️ No documents available for context.")
        return {**state, "generation": "I don't have enough context to answer the question."}

    try:
        # Generate answer using the QA chain
        generation = qa_rag_chain.invoke({
            "context": documents,
            "question": question
        })

        print("✅ Answer generated.")
        return {
            **state,
            "generation": generation
        }

    except Exception as e:
        print(f"❌ Error during answer generation: {str(e)}")
        return {
            **state,
            "generation": f"Error generating answer: {str(e)}"
        }

# =======================Build the Agent Graph with LangGraph=======================

# --- Build enhanced Agentic RAG with SQL-priority + Vector fallback + multi-hop merge ---
def create_rag_graph():
    from langgraph.graph import StateGraph, END

    agentic_rag = StateGraph(GraphState)

    # Add nodes
    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("query_sql", query_sql)
    agentic_rag.add_node("assess_combined_documents", assess_combined_documents)
    agentic_rag.add_node("rewrite_query", rewrite_query)
    agentic_rag.add_node("web_search", web_search)
    agentic_rag.add_node("generate_answer", generate_answer)

    # Entry point
    agentic_rag.set_entry_point("retrieve")

    # Flow: retrieve → grade_documents
    agentic_rag.add_edge("retrieve", "grade_documents")

    # Sau khi grade: Nếu docs tốt → generate_answer, nếu không → query_sql
    agentic_rag.add_conditional_edges(
        "grade_documents",
        lambda state: "generate_answer" if len(state.get("documents", [])) >= 1 else "query_sql",
        {
            "generate_answer": "generate_answer",
            "query_sql": "query_sql"
        }
    )

    agentic_rag.add_edge("query_sql", "assess_combined_documents")

    agentic_rag.add_conditional_edges(
        "assess_combined_documents",
        decide_after_reassessment,  # return "generate_answer" or "rewrite_query"
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query"
        }
    )

    # Web search flow
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")

    # END
    agentic_rag.add_edge("generate_answer", END)

    return agentic_rag.compile()
