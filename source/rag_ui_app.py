import gradio as gr
from PIL import Image
import base64
import io
import traceback
from langchain_core.documents import Document
from agentic_langgraph import create_rag_graph  # Hàm bạn đã xây
import os

# Format kết quả trả về
def format_response(response):
    try:
        docs = response.get("documents", [])
        generation = response.get("generation", "No answer available.")

        for doc in docs:
            if isinstance(doc, Document):
                image_base64 = doc.metadata.get("image_base64")
                if image_base64:
                    return generation, image_base64  # Trả về base64 ảnh nếu có

        return generation, None
    except Exception as e:
        return f"Error formatting response: {str(e)}", None

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
        print("✅ Response received.")

        generation, image_base64 = format_response(response)

        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))  # ✅ Decode về ảnh PIL
                return generation, image
            except Exception as e:
                return f"{generation}\n\n⚠️ Error decoding image: {str(e)}", None

        return generation, None

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None

# Giao diện Gradio
with gr.Blocks(title="Agentic RAG for Finance") as demo:
    gr.Markdown("# 💹 Financial Assistant – Agentic RAG")
    gr.Markdown("Ask questions about financial metrics, stock data, and get both textual and visual answers.")

    query_input = gr.Textbox(label="Enter your question", placeholder="e.g., Show Bollinger Bands for AAPL", lines=2)
    run_button = gr.Button("🧠 Get Answer")
    output_text = gr.Textbox(label="📜 Answer", lines=6)
    output_image = gr.Image(label="📈 Chart (if available)", type="pil")

    run_button.click(fn=rag_agent, inputs=[query_input], outputs=[output_text, output_image])

# Chạy app
if __name__ == "__main__":
    demo.launch(show_api=False)
