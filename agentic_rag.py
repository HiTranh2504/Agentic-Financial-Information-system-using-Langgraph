import os
import re
import json
import yaml
import pandas as pd
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
import gzip
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

from finance_metrics import load_formulas, identify_metric, get_required_fields, compute_metric
from plot_metric import plot_metric
from langchain.schema import Document


load_dotenv()

import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

##=================================BUILD A VECTOR DB FOR CSV DATA=================================##
from langchain_openai import OpenAIEmbeddings
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')


companies_df = pd.read_csv("Data Kien/djia_companies_20250426.csv")
companies_df.head()

docs = [
    Document(
        page_content=row['description'],
        metadata={'symbol': row['symbol'], 'name': row['name'], 'sector': row['sector']}
    )
    for _, row in companies_df.iterrows()
]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(docs)

embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

from langchain_chroma import Chroma

chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embed_model,
    collection_name='djia_company_info',
    persist_directory="./djia_vector_db"
)
similarity_threshold_retriever  = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.3})

##=================================Create a Query Retrieval Grader=================================##
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Parser
parser = StrOutputParser()

# Prompt system
SYS_PROMPT = """
You are an expert grader assessing relevance of a retrieved document to a user question.
Answer only 'yes' or 'no' depending on whether the document is relevant to the question.
"""

# Tạo prompt template
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT),
    ("human", "Retrieved document:\n{document}\n\nUser question:\n{question}")
])

# Tạo chain xử lý
doc_grader = grade_prompt | llm | parser

##=================================BUILD A QA RAG CHAIN=================================##
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


##=================================CREATE A QUERY REPHRASER=================================##
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


##=================================LOAD WEB SEARCH TOOL=================================##
from langchain_community.tools.tavily_search import TavilySearchResults
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)

##=================================SQL Query=================================##

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

##=================================BUILD AGENTIC RAG COMPONENT=================================##
# Graph state
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document

class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    use_sql: str
    documents: List[Document]

# Retrieve function for retrieval from Vector DB
def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "generation": "",
        "web_search_needed": "No",
        "use_sql": "No"
    }

# Grade documents 
def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"

    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.strip().lower()

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                # Phân loại theo nội dung
                if "database" in question.lower() or "truy vấn" in question.lower() or "from table" in question.lower():
                    use_sql = "Yes"
                    print("---GRADE: DOCUMENT NOT RELEVANT, SUGGEST SQL---")
                else:
                    web_search_needed = "Yes"
                    print("---GRADE: DOCUMENT NOT RELEVANT, SUGGEST WEB SEARCH---")
    else:
        # Kiểm tra nội dung câu hỏi → có chứa từ khóa SQL
        if "database" in question.lower() or "truy vấn" in question.lower() or "from table" in question.lower():
            use_sql = "Yes"
            print("---NO DOCS, BUT QUESTION SUGGESTS SQL---")
        else:
            web_search_needed = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed,
        "use_sql": use_sql,
        "generation": ""
    }

# rewrite query
def rewrite_query(state):
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question,
        "web_search_needed": state.get("web_search_needed", "No"),
        "use_sql": state.get("use_sql", "No"),
        "generation": ""
    }

# web search
def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    try:
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
    except Exception as e:
        print(f"Error during web search: {e}")
        web_results = Document(page_content=f"Error during web search: {str(e)}")
        documents.append(web_results)

    return {
        "documents": documents,
        "question": question,
        "web_search_needed": "No",
        "use_sql": state.get("use_sql", "No"),
        "generation": ""
    }

from langchain.schema import Document
import pandas as pd

def query_sql(state):
    print("---EXECUTE RAW SQL QUERY---")
    question = state["question"]

    # Kết nối đến DB
    conn = connect_to_database()
    if conn is None:
        raise ValueError("Không thể kết nối cơ sở dữ liệu.")

    # Sinh câu truy vấn SQL
    schema_info = get_schema_and_samples(conn)
    sql_query = generate_sql_query(question, schema_info)
    if not sql_query:
        conn.close()
        raise ValueError("Không thể sinh truy vấn SQL từ câu hỏi.")

    # Thực thi
    results = execute_sql_query(conn, sql_query)
    conn.close()

    # ✅ Xử lý kết quả
    if results is None or results.empty:
        content = "⚠️ Không có kết quả từ truy vấn SQL."
    else:
        content = (
            f"📊 Kết quả từ truy vấn SQL:\n\n{results.to_markdown(index=False)}"
        )

    # ✅ Gắn vào document và lưu lại SQL truy vấn nếu cần
    doc = Document(page_content=content)
    return {
        "documents": state["documents"] + [doc],
        "question": question,
        "sql_query": sql_query  # ⬅️ tùy chọn: nếu muốn dùng lại sau
    }


# Generate answer
def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    try:
        generation = qa_rag_chain.invoke({"context": documents, "question": question})
    except Exception as e:
        generation = f"Error generating answer: {str(e)}"
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "web_search_needed": state.get("web_search_needed", "No"),
        "use_sql": state.get("use_sql", "No")
    }

# Compute FA and TA
from langgraph.graph import END, StateGraph
from langchain.schema import Document
from finance_metrics import load_formulas, identify_metric, get_required_fields, compute_metric
from plot_metric import plot_metric
import pandas as pd

# --- Node: compute_fa ---
def compute_fa(state):
    question = state["question"]
    conn = connect_to_database()
    metadata = load_formulas()
    _, metric_name = identify_metric(question, metadata)
    required_fields = get_required_fields("FA", metric_name, metadata)

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

def compute_ta(state):
    question = state["question"]
    conn = connect_to_database()
    metadata = load_formulas()
    _, metric_name = identify_metric(question, metadata)

    df = pd.read_sql('SELECT "Date", "Close" FROM djia_prices WHERE "Ticker" = \'AAPL\' ORDER BY "Date" ASC LIMIT 100', conn)
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

# Decision node
def decide_branch(state):
    question = state["question"].lower()
    metadata = load_formulas()
    _, metric = identify_metric(question, metadata)

    if metric in ["RSI", "MACD", "MA", "BollingerBands"]:
        return "compute_ta"
    elif metric in ["EPS", "PE", "ROE", "DebtRatio"]:
        return "compute_fa"

    try:
        schema_info = get_schema_and_samples(connect_to_database())
        sql_query = generate_sql_query(question, schema_info)
        if sql_query:
            return "query_sql"
    except Exception:
        pass

    return "rewrite_query"


##=================================BUILD THE AGENT GRAPH WITH LANGGRAPH=================================##
from langgraph.graph import END, StateGraph
# --- Build the agent graph ---
agentic_rag = StateGraph(GraphState)

# Add nodes
agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)
agentic_rag.add_node("query_sql", query_sql)
agentic_rag.add_node("compute_ta", compute_ta)
agentic_rag.add_node("compute_fa", compute_fa)
agentic_rag.add_node("generate_answer", generate_answer)

# Set flow
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")

agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_branch,
    {
        "compute_ta": "compute_ta",
        "compute_fa": "compute_fa",
        "query_sql": "query_sql",
        "rewrite_query": "rewrite_query"
    }
)

agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("query_sql", "generate_answer")
agentic_rag.add_edge("compute_fa", "generate_answer")
agentic_rag.add_edge("compute_ta", "generate_answer")
agentic_rag.add_edge("generate_answer", END)

# Compile
agentic_rag = agentic_rag.compile()