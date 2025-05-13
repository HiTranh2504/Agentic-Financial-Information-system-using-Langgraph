import gradio as gr
from langchain.schema import Document
from agentic_rag import create_rag_graph  # Import the graph creation function
import base64
import traceback


def format_response(response):
    try:
        docs = response.get("documents", [])
        generation = response.get("generation", "Không có câu trả lời.")

        # Nếu có ảnh được đính kèm trong metadata
        for doc in docs:
            if isinstance(doc, Document):
                image_base64 = doc.metadata.get("image_base64")
                if image_base64:
                    return generation, image_base64

        return generation, None
    except Exception as e:
        print(f"Error formatting response: {str(e)}")
        return f"Lỗi khi xử lý phản hồi: {str(e)}", None


def rag_agent(query):
    try:
        # Create a new graph instance for each query
        agentic_rag = create_rag_graph()
        
        # Initialize state with all required fields
        initial_state = {
            "question": query,
            "generation": "",
            "web_search_needed": "No",
            "use_sql": "No",
            "documents": []
        }
        
        print(f"Processing query: {query}")
        response = agentic_rag.invoke(initial_state)
        print(f"Got response: {response}")
        
        generation, image_base64 = format_response(response)

        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                return generation, image_data
            except Exception as e:
                print(f"Error decoding image: {str(e)}")
                return f"{generation}\n\nLỗi khi xử lý hình ảnh: {str(e)}", None

        return generation, None

    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None


with gr.Blocks(title="Agentic RAG for Finance") as demo:
    gr.Markdown("# 📊 Financial Assistant Agentic RAG")
    gr.Markdown("Hỏi về dữ liệu chứng khoán, chỉ số phân tích cơ bản và kỹ thuật.")

    query_input = gr.Textbox(label="Câu hỏi của bạn", placeholder="Ví dụ: Bollinger Bands của cổ phiếu Apple trong 30 ngày gần đây")
    output_text = gr.Textbox(label="Câu trả lời", lines=4)
    output_image = gr.Image(label="Biểu đồ (nếu có)", type="pil")

    query_input.submit(fn=rag_agent, inputs=[query_input], outputs=[output_text, output_image])

if __name__ == "__main__":
    demo.launch()