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

# Rewrite query
def rewrite_query(state):
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# Query SQL

def query_sql(state):
    print("---EXECUTE RAW SQL QUERY OR METRIC COMPUTATION---")
    question = state["question"]
    metadata = load_formulas()
    category, metric_name = identify_metric(question, metadata)

    # Kết nối đến DB
    conn = connect_to_database()
    if conn is None:
        raise ValueError("Không thể kết nối cơ sở dữ liệu.")

    # Nếu là chỉ số tài chính (FA hoặc TA)
    if metric_name:
        print(f"📌 Nhận diện chỉ số: {metric_name} ({category})")

        # --- TA: Vẽ biểu đồ từ dữ liệu giá ---
        if category == "TA":
            df = pd.read_sql(
                'SELECT "Date", "Close" FROM djia_prices WHERE "Ticker" = \'AAPL\' ORDER BY "Date" ASC LIMIT 100',
                conn
            )
            conn.close()
            df.rename(columns={"Date": "date", "Close": "price"}, inplace=True)

            image_base64 = plot_metric(metric_name, df)
            result_doc = Document(
                page_content=f"Biểu đồ {metric_name} cho AAPL:",
                metadata={"image_base64": image_base64}
            )
            return {
                "documents": state["documents"] + [result_doc],
                "question": question
            }

        # --- FA: Tính toán chỉ số từ bảng công ty ---
        elif category == "FA":
            required_fields = get_required_fields(category, metric_name, metadata)
            query = f"SELECT {', '.join(required_fields)} FROM djia_companies WHERE symbol = 'AAPL' LIMIT 1;"
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

    # Trường hợp không phải chỉ số → xử lý như SQL tự nhiên
    schema_info = get_schema_and_samples(conn)
    sql_query = generate_sql_query(question, schema_info)
    if not sql_query:
        conn.close()
        raise ValueError("Không thể sinh truy vấn SQL từ câu hỏi.")

    results = execute_sql_query(conn, sql_query)
    conn.close()

    # Nếu không có dữ liệu → yêu cầu fallback web search
    if results is None or results.empty:
        content = "⚠️ Không có kết quả từ truy vấn SQL."
        state["web_search_needed"] = "Yes"
    else:
        content = f"📊 Kết quả từ truy vấn SQL:\n\n{results.to_markdown(index=False)}"

    doc = Document(page_content=content)

    return {
        "documents": state["documents"] + [doc],
        "question": question,
        "sql_query": sql_query,
        "web_search_needed": state.get("web_search_needed", "No")  # vẫn duy trì trạng thái fallback
    }


# Decide merge or query sql
def decide_merge_or_sql(state):
    docs = state.get("documents", [])
    if len(docs) >= 1:
        return "merge_documents"
    return "query_sql"


# Merge documents
def merge_documents(state):
    vector_docs = state.get("documents", [])
    sql_doc = state.get("sql_result", None)

    if sql_doc:
        merged_docs = vector_docs + [sql_doc]
    else:
        merged_docs = vector_docs

    return {
        **state,
        "documents": merged_docs
    }


# Decide web search or generate answer
def decide_after_sql(state):
    if state.get("web_search_needed", "No") == "Yes":
        print("---DECISION: Missing SQL data → web search---")
        return "rewrite_query"
    
    print("---DECISION: Sufficient SQL data → generate answer---")
    return "generate_answer"


# Web search
def web_search(state):
    """
    Web search based on the re-written question.
    Args:
    state (dict): The current graph state
    Returns:
    state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = tv_search.invoke(question)
    
    # Gỡ lỗi: kiểm tra cấu trúc trả về
    print("DOCS TYPE:", type(docs))
    print("FIRST ELEMENT TYPE:", type(docs[0]) if docs else "EMPTY")

    # Nếu docs là list of strings
    if isinstance(docs[0], str):
        web_content = "\n\n".join(docs)
    # Nếu docs là list of dicts
    elif isinstance(docs[0], dict) and "content" in docs[0]:
        web_content = "\n\n".join([d["content"] for d in docs])
    # Nếu docs là list of Document
    elif isinstance(docs[0], Document):
        web_content = "\n\n".join([d.page_content for d in docs])
    else:
        raise TypeError("Unsupported doc format")

    web_results = Document(page_content=web_content)
    documents.append(web_results)

    return {"documents": documents, "question": question}


# Generate Answer
def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question,"generation": generation}


# =======================Build the Agent Graph with LangGraph=======================

# --- Build enhanced Agentic RAG with SQL-priority + Vector fallback + multi-hop merge ---
def create_rag_graph():
    agentic_rag = StateGraph(GraphState)

    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("merge_documents", merge_documents)
    agentic_rag.add_node("query_sql", query_sql)
    agentic_rag.add_node("rewrite_query", rewrite_query)
    agentic_rag.add_node("web_search", web_search)
    agentic_rag.add_node("generate_answer", generate_answer)

    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")

    agentic_rag.add_conditional_edges("grade_documents", decide_merge_or_sql, {
        "merge_documents": "merge_documents",
        "query_sql": "query_sql"
    })

    agentic_rag.add_edge("merge_documents", "query_sql")

    agentic_rag.add_conditional_edges("query_sql", decide_after_sql, {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query"
    })

    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)

    return agentic_rag.compile()
