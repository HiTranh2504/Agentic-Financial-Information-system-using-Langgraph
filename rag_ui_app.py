import gradio as gr
from langchain.schema import Document
from agentic_rag import agentic_rag  # Đảm bảo bạn đã import đúng graph
import base64


def format_response(response):
    docs = response.get("documents", [])
    generation = response.get("generation", "Không có câu trả lời.")

    # Nếu có ảnh được đính kèm trong metadata
    for doc in docs:
        image_base64 = doc.metadata.get("image_base64") if isinstance(doc, Document) else None
        if image_base64:
            return generation, image_base64

    return generation, None


def rag_agent(query):
    try:
        response = agentic_rag.invoke({"question": query})
        generation, image_base64 = format_response(response)

        if image_base64:
            image_data = base64.b64decode(image_base64)
            return generation, image_data

        return generation, None

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", None


with gr.Blocks(title="Agentic RAG for Finance") as demo:
    gr.Markdown("# 📊 Financial Assistant Agentic RAG")
    gr.Markdown("Hỏi về dữ liệu chứng khoán, chỉ số phân tích cơ bản và kỹ thuật.")

    query_input = gr.Textbox(label="Câu hỏi của bạn", placeholder="Ví dụ: Bollinger Bands của cổ phiếu Apple trong 30 ngày gần đây")
    output_text = gr.Textbox(label="Câu trả lời", lines=4)
    output_image = gr.Image(label="Biểu đồ (nếu có)", type="pil")

    query_input.submit(fn=rag_agent, inputs=[query_input], outputs=[output_text, output_image])

if __name__ == "__main__":
    demo.launch()