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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter



load_dotenv()

import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

##=================================BUILD A VECTOR DB FOR WEKIPEDIA DATA=================================##
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

wikipedia_filepath = 'D:\HK2_2024_2025\Data Platform\Thuc_hanh\CK\chatbot-finacial-langgraph\data\simplewiki-2020-11-01.jsonl.gz'
docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        # Add documents
        docs.append({
            'metadata': {
                'title': data.get('title'),
                'article_id': data.get('id')
            },
            'data': ' '.join(data.get('paragraphs')[0:3])  # restrict data to first 3 paragraphs to run later modules faster
            })
docs = [doc for doc in docs for x in ['india'] if x in doc['data'].lower().split()]
docs = [Document(page_content=doc['data'], metadata=doc['metadata']) for doc in docs]
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)

chroma_db = Chroma.from_documents(documents=chunked_docs, collection_name='rag_wikipedia_db',embedding=openai_embed_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="./wikipedia_db")

# set up a vector database retriever
similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.3})

# Create a query retrieval chain
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
# Load Web Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000)

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
            cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
            sample_rows = cursor.fetchall()
            colnames = [desc[0] for desc in cursor.description]
            schema_info[f"{table}_samples"] = [dict(zip(colnames, row)) for row in sample_rows]
        cursor.close()
        return schema_info
    except Exception as e:
        return {"error": str(e)}

def load_metadata():
    try:
        with open("metadata.yml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

def generate_sql_query(user_question, schema_info=None):
    prompt = f"""
You are a PostgreSQL expert working with a single table called `stocks`.

This table contains daily stock information with the following columns:

- date: Date of the record (format: YYYY-MM-DD)
- price: The closing price of the stock on that date (FLOAT)
- open: The opening price of the stock on that date (FLOAT)
- high: The highest price of the stock on that date (FLOAT)
- low: The lowest price of the stock on that date (FLOAT)
- volume: Trading volume (format: float + 'M' suffix for millions)
- change_percent: Daily percentage change in stock price (can be positive or negative)

Assume this table is already in a PostgreSQL database as `stocks`.

User Question:
{user_question}

Write a correct and optimized PostgreSQL query to answer this question. 
Do not explain or wrap the query in markdown. Just return raw SQL.
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
        print(f"Lỗi sinh SQL: {e}")
        return None

def execute_sql_query(conn, query):
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Lỗi thực thi SQL: {e}")
        return None
    
def run_chat(question: str):
    conn = connect_to_database()
    if conn is None:
        return

    schema_info = get_schema_and_samples(conn)
    sql_query = generate_sql_query(question, schema_info)

    if not sql_query:
        print("Không thể sinh truy vấn SQL.")
        return

    print(f"\nSQL sinh ra:\n{sql_query}")
    results = execute_sql_query(conn, sql_query)

    if results is None:
        print("Truy vấn lỗi.")
        return

    print("\nDữ liệu:")
    print(results.head(5))

    conn.close()

##=================================BUILD AGENTIC RAG COMPONENT=================================##
# Graph state
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
	# Retrieval
	documents = similarity_threshold_retriever.invoke(question)
	return {"documents": documents, "question": question}


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
		"use_sql": use_sql
	}

# rewrite query
def rewrite_query(state):
    """
    Rewrite the query to produce a better question.
    Args:
    state (dict): The current graph state
    Returns:
    state (dict): Updates question key with a re-phrased or re-written question
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# web search
from langchain.schema import Document
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


# Use SQL
from finance_metrics import load_formulas, identify_metric, get_required_fields, compute_metric
from plot_metric import plot_metric
from langchain.schema import Document
import pandas as pd


def query_sql(state):
    print("---EXECUTE SQL QUERY OR METRIC COMPUTATION---")
    question = state["question"]

    # Kết nối đến cơ sở dữ liệu
    conn = connect_to_database()
    if conn is None:
        raise ValueError("Không thể kết nối cơ sở dữ liệu.")

    schema_info = get_schema_and_samples(conn)
    metadata = load_formulas()
    category, metric_name = identify_metric(question, metadata)

    # Nếu là câu hỏi về chỉ số tài chính
    if metric_name:
        print(f"Phát hiện truy vấn liên quan đến chỉ số: {metric_name}")
        required_fields = get_required_fields(category, metric_name, metadata)

        # TA metrics - cần vẽ biểu đồ
        if metric_name in ["RSI", "MACD", "MA", "BollingerBands"]:
            df = pd.read_sql("SELECT date, price FROM stocks ORDER BY date ASC LIMIT 100", conn)
            conn.close()

            image_base64 = plot_metric(metric_name, df)
            result_doc = Document(
                page_content=f"Biểu đồ {metric_name}:",
                metadata={"image_base64": image_base64}
            )

            return {
                "documents": state["documents"] + [result_doc],
                "question": question
            }

        # FA metrics - tính toán dựa trên 1 dòng dữ liệu
        else:
            field_clause = ", ".join(required_fields)
            query = f"SELECT {field_clause} FROM stocks LIMIT 1;"
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                result_str = f"Không tìm thấy đủ dữ liệu để tính {metric_name}."
            else:
                row = df.iloc[0].to_dict()
                result = compute_metric(metric_name, row)
                result_str = f"{metric_name} = {result}"

            return {
                "documents": state["documents"] + [Document(page_content=result_str)],
                "question": question
            }

    # Truy vấn SQL thông thường
    sql_query = generate_sql_query(question, schema_info)
    if not sql_query:
        raise ValueError("Không thể sinh truy vấn SQL từ câu hỏi.")

    results = execute_sql_query(conn, sql_query)
    conn.close()

    if results is None or results.empty:
        content = "Không có kết quả từ truy vấn SQL."
    else:
        content = results.to_markdown(index=False)  # hoặc `.to_string(index=False)` nếu không muốn bảng Markdown

    # Gán vào document để các bước sau có thể sinh câu trả lời
    doc = Document(page_content=content)

    # Cập nhật state
    return {
        "documents": state["documents"] + [doc],
        "question": question
    }


# Generate answer
def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question,"generation": generation}


# decide to generate
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state.get("web_search_needed", "No")
    use_sql = state.get("use_sql", "No")

    if use_sql == "Yes":
        print("---DECISION: QUERY SQL DATABASE---")
        return "query_sql"  # Bạn cần tạo node mới: `query_sql`

    elif web_search_needed == "Yes":
        print("---DECISION: WEB SEARCH NEEDED, REWRITE QUERY---")
        return "rewrite_query"

    else:
        print("---DECISION: DOCUMENTS ARE RELEVANT, GENERATE ANSWER---")
        return "generate_answer"


##=================================BUILD THE AGENT GRAPH WITH LANGGRAPH=================================##
from langgraph.graph import END, StateGraph
agentic_rag = StateGraph(GraphState)
# Define the nodes
agentic_rag.add_node("retrieve", retrieve) # retrieve
agentic_rag.add_node("grade_documents", grade_documents) # grade documents
agentic_rag.add_node("rewrite_query", rewrite_query) # transform_query
agentic_rag.add_node("web_search", web_search) # web search
agentic_rag.add_node("query_sql", query_sql) 
agentic_rag.add_node("generate_answer", generate_answer) # generate answer
# Build graph
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
# Quyết định nhánh tiếp theo sau khi chấm điểm tài liệu
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_query": "rewrite_query",
        "generate_answer": "generate_answer",
        "query_sql": "query_sql"
    }
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("query_sql", "generate_answer")
agentic_rag.add_edge("generate_answer", END)
# Compile
agentic_rag = agentic_rag.compile()