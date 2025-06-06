# 🧠 FinAgent: A LangGraph-Orchestrated Agentic RAG System for Financial Question Answering

---

## 📌 Abstract

**FinAgent** is a modular, intelligent system for automated **financial question answering (QA)**. It combines advanced AI techniques such as:

- **Retrieval-Augmented Generation (RAG)**
- **SQL-based structured querying**
- **Semantic vector search using FAISS**
- **Dynamic chart generation**
- **Web search fallback**

All components are orchestrated through **LangGraph**, enabling agentic reasoning and adaptive query routing. The system is designed to answer factual and analytical queries using structured financial databases (e.g., DJIA prices), vectorized financial documents, and real-time search when necessary.

---

## 🎯 Motivation

Most QA systems are specialized in either structured data (SQL) or unstructured text (RAG). FinAgent bridges both by:

- Allowing **accurate metric computation** from financial databases  
- Providing **semantic retrieval** from regulatory filings (e.g., SEC documents)  
- Supporting **chart generation** (line, heatmap, boxplot, etc.)  
- Using **web search fallback** when data is missing or incomplete  

The hybrid architecture ensures **accuracy**, **explainability**, and **coverage** — crucial for financial applications.

---

## 🔍 Key Capabilities

| **Component**             | **Description**                                                                |
|---------------------------|--------------------------------------------------------------------------------|
| `LangGraph DAG`           | Controls multi-step query routing with conditional logic                      |
| `SQL Generator (LLM)`     | Converts natural questions into valid PostgreSQL queries                      |
| `Chart Detector`          | Detects and renders charts from financial data                                |
| `FAISS Vector Search`     | Performs semantic retrieval over embedded financial texts                     |
| `LLM Grader`              | Validates the relevance of documents using GPT-4-based scoring                 |
| `Web Search Agent`        | Rewrites vague questions and executes fallback search using Tavily API        |

---

## 🧠 System Architecture

```mermaid
graph TD
  A[User Question] --> B[Vector DB Retrieval]
  B --> C[LLM Grading]
  C -->|Relevant| D[LLM Answer Generator]
  C -->|Irrelevant| E[SQL Query Generator]
  E --> F[Execute SQL]
  F --> G[Chart Detection]
  G -->|Chart Required| H[Chart Generator]
  G -->|No Chart| D
  H --> D
  F -->|Empty| I[Rewrite → Web Search]
  I --> D
