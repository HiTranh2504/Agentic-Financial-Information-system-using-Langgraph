import gradio as gr
import base64
import traceback
from langchain_core.documents import Document
from agentic_langgraph import create_rag_graph  # Import graph tạo sẵn
from PIL import Image
import io


# Format kết quả trả về
def format_response(response):
    try:
        docs = response.get("documents", [])
        generation = response.get("generation", "Không có câu trả lời.")

        for doc in docs:
            if isinstance(doc, Document):
                image_base64 = doc.metadata.get("image_base64")
                if image_base64:
                    return generation, image_base64

        return generation, None
    except Exception as e:
        return f"Lỗi khi xử lý phản hồi: {str(e)}", None


# Hàm chính chạy graph
def rag_agent(query):
    try:
        agentic_rag = create_rag_graph()

        initial_state = {
            "question": query,
            "generation": "",
            "web_search_needed": "No",
            "documents": []
        }

        print(f"Processing query: {query}")
        response = agentic_rag.invoke(initial_state)
        print("Response received.")

        generation, image_base64 = format_response(response)

        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                return generation, image
            except Exception as e:
                return f"{generation}\n\nLỗi xử lý ảnh: {str(e)}", None

        return generation, None

    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None


# Giao diện Gradio
with gr.Blocks(title="Agentic RAG for Finance") as demo:
    gr.Markdown("# 💹 Trợ lý Tài chính - Agentic RAG")
    gr.Markdown("Bạn có thể hỏi về các chỉ số tài chính, dữ liệu cổ phiếu, và phân tích kỹ thuật/phân tích cơ bản.")

    query_input = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: PE ratio của Apple", lines=2)
    run_button = gr.Button("🧠 Trả lời")
    output_text = gr.Textbox(label="📜 Câu trả lời", lines=6)
    output_image = gr.Image(label="📈 Biểu đồ (nếu có)", type="pil")

    run_button.click(fn=rag_agent, inputs=[query_input], outputs=[output_text, output_image])

# Chạy app
if __name__ == "__main__":
    demo.launch(show_api=False)
