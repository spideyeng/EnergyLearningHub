"""
Energy Learning Hub — RAG-Powered Knowledge System
===================================================
Fixed for Render deployment:
  - Binds to PORT immediately (Render default: 10000)
  - Loads RAG pipeline in background thread after server starts
  - Shows loading state in UI until pipeline is ready
  - Uses API-based embeddings (Gemini) to avoid loading PyTorch locally
  - Frees intermediate data after pipeline init to reduce memory footprint
"""

import os
import glob
import threading
import time
import gradio as gr
from operator import itemgetter

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION                                           ║
# ╚══════════════════════════════════════════════════════════════╝
PDF_DIR = os.environ.get("PDF_DIR", "./data")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
PORT = int(os.environ.get("PORT", 10000))  # Render default is 10000

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. GLOBAL STATE — populated by background thread           ║
# ╚══════════════════════════════════════════════════════════════╝
pipeline_ready = False
pipeline_status = "Initializing..."
qa_chain = None
llm_with_fallback = None
ingestion_log = []
first_chunk_previews = {}
total_chunks = 0
total_vectors = 0


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. FAQ KNOWLEDGE BASE                                      ║
# ╚══════════════════════════════════════════════════════════════╝
FAQ_DATABASE = {
    "beginner": {
        "label": "🟢 Beginner — Foundations",
        "description": "Start here if you're new to energy concepts.",
        "categories": {
            "Energy Basics ⚡": [
                {"q": "What is the difference between oil, natural gas, and coal?", "jumpstart": "Oil is a liquid fossil fuel used mainly for transportation and petrochemicals. Natural gas is gaseous, often used for heating and electricity. Coal is solid, historically used for power generation and steelmaking. Each has different carbon intensity and energy density.", "sources": ["Introductory energy textbooks", "IEA/EIA primers"]},
                {"q": "How is energy measured (joules, BTUs, kWh)?", "jumpstart": "Energy is measured in joules (J), British Thermal Units (BTUs), or kilowatt-hours (kWh). For example, 1 kWh powers a 100-watt light bulb for 10 hours.", "sources": ["General science references", "EIA energy glossary"]},
                {"q": "What are primary vs. secondary energy sources?", "jumpstart": "Primary sources (coal, crude oil, sunlight) exist in nature. Secondary sources (electricity, gasoline) are converted from primary sources for practical use.", "sources": ["Introductory energy textbooks"]}
            ],
            "Energy Systems 🔌": [
                {"q": "How is electricity generated from fossil fuels?", "jumpstart": "Fossil fuels are burned to heat water, producing steam that spins turbines connected to generators. This mechanical energy is converted into electricity.", "sources": ["Power systems textbooks", "EIA resources"]},
                {"q": "What are upstream, midstream, and downstream operations?", "jumpstart": "Upstream covers exploration and production. Midstream involves transportation and storage (pipelines, shipping). Downstream refers to refining and distribution into usable products like gasoline.", "sources": ["Industry primers", "SPE resources"]},
                {"q": "What are renewable energy sources?", "jumpstart": "Renewables include solar, wind, hydro, and geothermal. They rely on naturally replenished resources and generally produce lower greenhouse gas emissions.", "sources": ["IRENA reports", "Introductory energy textbooks"]}
            ],
            "Careers & Skills 🎓": [
                {"q": "What skills are needed to work in the energy industry?", "jumpstart": "Technical skills (engineering, geology, data analysis), business skills (economics, project management), and increasingly sustainability knowledge are valuable.", "sources": ["SPE career guides", "AAPG resources"]}
            ]
        },
        "next_steps": {"Want technical detail?": "intermediate", "Want policy & global view?": "advanced"}
    },
    "intermediate": {
        "label": "🟡 Intermediate — Industry Insights",
        "description": "For learners who understand the basics and want to explore operations and markets.",
        "categories": {
            "Resources & Operations 🛢️": [
                {"q": "What are proven reserves and how are they estimated?", "jumpstart": "Proven reserves are quantities of oil or gas that geological and engineering data show can be recovered under current economic and operating conditions.", "sources": ["Petroleum geology textbooks", "SPE Reserves Definitions"]},
                {"q": "What is the difference between baseload and peak load power?", "jumpstart": "Baseload power is the steady supply needed 24/7 (often from coal, nuclear, or hydro). Peak load power meets demand spikes (often from gas turbines or renewables with storage).", "sources": ["Electricity markets textbooks", "Grid operations guides"]}
            ],
            "Markets & Economics 📈": [
                {"q": "How are oil prices determined?", "jumpstart": "Prices are influenced by supply and demand, geopolitical events, OPEC decisions, production costs, and futures trading. They can be highly volatile.", "sources": ["BP Statistical Review", "Market analysis reports"]},
                {"q": "What is the role of OPEC in global energy markets?", "jumpstart": "OPEC coordinates oil production policies among member states to influence global supply and stabilize prices.", "sources": ["OPEC annual reports", "Energy outlook publications"]}
            ],
            "Sustainability & Security 🌱": [
                {"q": "What is carbon capture and storage (CCS)?", "jumpstart": "CCS captures CO2 emissions from power plants or industrial sites and stores them underground to reduce atmospheric emissions.", "sources": ["CCS textbooks", "IEA CCS reports"]},
                {"q": "What is energy security and why is it important?", "jumpstart": "Energy security means ensuring reliable, affordable access to energy through diversifying sources, securing supply chains, and reducing import dependence.", "sources": ["IEA energy security publications", "Policy reports"]},
                {"q": "Which professional certifications are useful?", "jumpstart": "Certifications like PMP (Project Management), CFA (energy finance), or specialized petroleum engineering courses can boost career prospects.", "sources": ["Professional certification syllabi", "SPE Handbook"]}
            ]
        },
        "next_steps": {"Want market & geopolitics?": "advanced", "Want technical engineering?": "advanced"}
    },
    "advanced": {
        "label": "🔵 Advanced — Strategic & Global Perspectives",
        "description": "For learners ready to engage with complex policy, geopolitics, and future trends.",
        "categories": {
            "Policy & Regulation 🏛️": [
                {"q": "How do international agreements (like the Paris Accord) shape energy policy?", "jumpstart": "Agreements like the Paris Accord push countries to reduce emissions, invest in renewables, and set long-term climate targets that reshape national energy strategies.", "sources": ["UNFCCC publications", "IEA policy reports"]},
                {"q": "What are net-zero targets and how do they affect the energy transition?", "jumpstart": "Net-zero means balancing CO2 emissions with removals. These targets drive investment in renewables, CCS, and efficiency improvements across the energy sector.", "sources": ["World Bank reports", "IEA Net Zero Roadmap"]}
            ],
            "Geopolitics & Strategy 🌐": [
                {"q": "How do geopolitics influence oil and gas markets?", "jumpstart": "Political tensions, sanctions, trade agreements, and regional conflicts can disrupt supply chains, shift trade flows, and cause price volatility in global energy markets.", "sources": ["Energy geopolitics texts", "The Prize by Daniel Yergin"]},
                {"q": "What is the impact of subsidies and regulation on energy markets?", "jumpstart": "Subsidies can lower consumer costs but distort markets. Regulations set safety and environmental standards. Together, they shape investment decisions and energy mix.", "sources": ["Energy economics journals", "World Bank policy papers"]}
            ],
            "Future of Energy 🚀": [
                {"q": "How does the energy transition affect traditional oil and gas companies?", "jumpstart": "Many companies are diversifying into renewables, hydrogen, and CCS. The transition creates both risks (stranded assets) and opportunities (new markets).", "sources": ["Corporate sustainability reports", "IEA World Energy Outlook"]},
                {"q": "How do grids and transmission systems work at scale?", "jumpstart": "Power grids balance generation and demand in real-time across vast networks. Modern grids integrate renewables, storage, and smart technology for reliability.", "sources": ["Power systems engineering texts", "Grid integration studies"]},
                {"q": "What are the major career pathways in renewables vs. fossil fuels?", "jumpstart": "Fossil fuel careers focus on geology, drilling, and refining. Renewable careers span solar/wind engineering, grid integration, policy, and energy storage. Many skills transfer between sectors.", "sources": ["Workforce trend studies", "Professional association guides"]}
            ]
        },
        "next_steps": {}
    }
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. BACKGROUND INITIALIZATION                               ║
# ╚══════════════════════════════════════════════════════════════╝
def initialize_pipeline():
    """Heavy initialization runs in a background thread AFTER the server binds to PORT."""
    global pipeline_ready, pipeline_status, qa_chain, llm_with_fallback
    global ingestion_log, first_chunk_previews, total_chunks, total_vectors

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_chroma import Chroma

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_KEY
        )

        # --- Try loading existing vector store from disk first ---
        vector_store = None
        if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            pipeline_status = "Loading existing vector store from disk..."
            print("💾 Found existing ChromaDB on disk, loading...")
            try:
                vector_store = Chroma(
                    collection_name="energy_hub",
                    embedding_function=embeddings,
                    persist_directory=CHROMA_DIR
                )
                existing_count = vector_store._collection.count()
                if existing_count > 0:
                    total_vectors = existing_count
                    print(f"✅ Loaded existing vector store: {total_vectors} vectors (skipping re-embedding)")
                else:
                    vector_store = None  # Empty store, rebuild
                    print("⚠️ Existing store is empty, will rebuild...")
            except Exception as e:
                vector_store = None
                print(f"⚠️ Could not load existing store ({e}), will rebuild...")

        # --- Build vector store from PDFs if not loaded from disk ---
        if vector_store is None:
            pipeline_status = "Loading PDFs..."
            print("📖 Loading PDFs from:", PDF_DIR)

            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            pdf_files = glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True)
            all_docs = []

            for pdf_path in sorted(pdf_files):
                filename = os.path.basename(pdf_path)
                filesize_kb = os.path.getsize(pdf_path) / 1024
                filesize_mb = filesize_kb / 1024
                try:
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load()
                    for page in pages:
                        page.metadata["source_file"] = filename
                        page.metadata["page_number"] = page.metadata.get("page", "N/A")
                        first_line = page.page_content.strip().split("\n")[0][:100]
                        if any(kw in first_line.lower() for kw in ["chapter", "part", "section", "appendix"]):
                            page.metadata["chapter"] = first_line
                        else:
                            page.metadata["chapter"] = "N/A"
                    all_docs.extend(pages)
                    ingestion_log.append({"filename": filename, "filesize": f"{filesize_mb:.2f} MB" if filesize_mb >= 1 else f"{filesize_kb:.1f} KB", "pages": len(pages), "status": "✅ Loaded"})
                    print(f"  ✅ {filename}: {len(pages)} pages")
                except Exception as e:
                    ingestion_log.append({"filename": filename, "filesize": "?", "pages": 0, "status": f"❌ {e}"})
                    print(f"  ❌ {filename}: {e}")

            # --- Chunk (larger chunks = fewer API calls) ---
            pipeline_status = "Chunking documents..."
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""])
            chunks = splitter.split_documents(all_docs)
            total_chunks = len(chunks)

            for chunk in chunks:
                src = chunk.metadata.get("source_file", "unknown")
                if src not in first_chunk_previews:
                    first_chunk_previews[src] = chunk.page_content[:300]

            print(f"🧩 Total chunks: {total_chunks}")

            # --- Batched embedding with retry (free tier: 100/min, 1000/day) ---
            pipeline_status = "Embedding chunks..."
            BATCH_SIZE = 80
            num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_num = i // BATCH_SIZE + 1
                batch = chunks[i : i + BATCH_SIZE]
                pipeline_status = f"Embedding batch {batch_num}/{num_batches} ({len(batch)} chunks)..."
                print(f"📦 Embedding batch {batch_num}/{num_batches} ({len(batch)} chunks)...")

                # Retry with exponential backoff for rate limits
                for attempt in range(5):
                    try:
                        if vector_store is None:
                            vector_store = Chroma.from_documents(
                                documents=batch,
                                collection_name="energy_hub",
                                embedding=embeddings,
                                persist_directory=CHROMA_DIR
                            )
                        else:
                            vector_store.add_documents(batch)
                        break  # Success
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            wait = 60 * (2 ** attempt)  # 60s, 120s, 240s, ...
                            print(f"⚠️ Rate limited, retrying in {wait}s (attempt {attempt + 1}/5)...")
                            pipeline_status = f"Rate limited — retrying batch {batch_num} in {wait}s..."
                            time.sleep(wait)
                        else:
                            raise

                # Rate-limit pause between batches (skip after last)
                if i + BATCH_SIZE < len(chunks):
                    print(f"⏳ Pausing 60s for rate limit...")
                    time.sleep(60)

            # Free intermediate data
            del all_docs
            del chunks
            import gc
            gc.collect()
            print("🧹 Freed intermediate data from memory.")

        total_vectors = vector_store._collection.count()
        print(f"✅ Vector store: {total_vectors} vectors")

        # --- LLMs ---
        pipeline_status = "Configuring LLMs..."
        print("🤖 Configuring LLMs...")

        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_openai import ChatOpenAI

        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GEMINI_KEY
        )

        # Qwen via HuggingFace's OpenAI-compatible Inference API
        # (no torch/transformers needed — just an HTTP call)
        qwen_llm = ChatOpenAI(
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=HF_TOKEN,
            temperature=0.1,
            max_tokens=1024,
        )
        llm_with_fallback = gemini_llm.with_fallbacks([qwen_llm])

        # --- RAG Chain ---
        pipeline_status = "Building RAG chain..."

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        template = """You are an expert energy educator for a learning hub that teaches
newcomers about oil, gas, and the broader energy sector. Answer the question
using ONLY the provided context from the knowledge base.

Guidelines:
- Use clear, accessible language appropriate for the learner's level
- Include specific data, examples, or case studies from the context when available
- If the context doesn't contain enough information, say so honestly
- ALWAYS cite your sources using the format [Source: filename, Page X]
- End with a suggestion for what to explore next

Context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        qa_chain = (
            {
                "context": itemgetter("query") | retriever,
                "question": itemgetter("query")
            }
            | prompt
            | llm_with_fallback
            | StrOutputParser()
        )

        pipeline_ready = True
        pipeline_status = "Ready"
        print("✅ Pipeline ready! Server is live.")

    except Exception as e:
        pipeline_status = f"Error: {str(e)}"
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. HELPER FUNCTIONS                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def get_model_source(response_obj):
    metadata = getattr(response_obj, "response_metadata", {})
    if "safety_ratings" in metadata or "prompt_feedback" in metadata:
        return "Gemini 2.5 Flash"
    return "Qwen 2.5 (HuggingFace Fallback)"


def find_jumpstart(question, level=None):
    q_lower = question.lower().strip()
    for lvl, data in FAQ_DATABASE.items():
        if level and lvl != level:
            continue
        for cat, faqs in data["categories"].items():
            for faq in faqs:
                if faq["q"].lower().strip() == q_lower:
                    return {"answer": faq["jumpstart"], "sources": faq["sources"], "level": lvl, "category": cat}
    return None


def get_faq_list(level_key):
    if level_key not in FAQ_DATABASE:
        return "Select a level to see FAQs."
    data = FAQ_DATABASE[level_key]
    lines = ["### " + data['label'], "_" + data['description'] + "_"]
    for cat, faqs in data["categories"].items():
        lines.append("\n**" + cat + "**")
        for faq in faqs:
            lines.append("- " + faq['q'])
    if data["next_steps"]:
        lines.append("\n---\n**👉 Where to go next:**")
        for label, target in data["next_steps"].items():
            lines.append("- " + label + " → **" + target.capitalize() + "** level")
    return "\n".join(lines)


def get_ingestion_summary():
    if not pipeline_ready:
        return "⏳ **Pipeline is loading:** " + pipeline_status + "\n\nRefresh this tab in a moment."
    if not ingestion_log:
        return "No PDFs found in the data directory."
    lines = ["### 📚 Ingested Documents\n", "| File | Size | Pages | Status |", "|------|------|-------|--------|"]
    for entry in ingestion_log:
        lines.append("| " + entry["filename"] + " | " + entry["filesize"] + " | " + str(entry["pages"]) + " | " + entry["status"] + " |")
    lines.append("\n**Total chunks:** " + str(total_chunks))
    lines.append("**Vector store:** " + str(total_vectors) + " vectors")
    if first_chunk_previews:
        lines.append("\n---\n### 🔍 First Chunk Preview\n")
        for src, preview in first_chunk_previews.items():
            lines.append("**" + src + ":**")
            lines.append("> " + preview[:250].replace("\n", " ") + "...\n")
    return "\n".join(lines)


def query_hub(user_query, selected_level):
    """Main query handler."""
    from langchain_core.messages import HumanMessage

    if not pipeline_ready:
        waiting = "⏳ **The knowledge base is still loading.** Status: " + pipeline_status + "\n\nPlease try again in a minute."
        return waiting, waiting, waiting

    level_map = {"All Levels": None, "🟢 Beginner": "beginner", "🟡 Intermediate": "intermediate", "🔵 Advanced": "advanced"}
    level_filter = level_map.get(selected_level)

    # Jump-start
    js = find_jumpstart(user_query, level_filter)
    if js:
        jumpstart_md = ("📋 **Jump-Start Answer** (" + js['level'].capitalize() + " | " + js['category'] + ")"
            + "\n\n" + js['answer'] + "\n\n📚 Suggested sources: " + ', '.join(js['sources']))
    else:
        jumpstart_md = "_No matching FAQ — showing RAG retrieval only._"

    # RAG
    try:
        rag_response = qa_chain.invoke({"query": user_query})
        rag_md = "📚 **RAG Deep-Dive Answer**" + "\n\n" + str(rag_response)
    except Exception as e:
        rag_md = "❌ RAG Error: " + str(e)

    # Direct LLM
    try:
        llm_response = llm_with_fallback.invoke([HumanMessage(content=user_query)])
        model_name = get_model_source(llm_response)
        llm_md = "🤖 [Source: " + model_name + "]" + "\n\n" + llm_response.content
    except Exception as e:
        llm_md = "❌ LLM Error: " + str(e)

    return jumpstart_md, rag_md, llm_md


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. GRADIO UI — Launches IMMEDIATELY to bind the port       ║
# ╚══════════════════════════════════════════════════════════════╝
with gr.Blocks(title="Energy Learning Hub") as demo:

    gr.Markdown(
        "# 🌍 Energy Learning Hub\n"
        "**RAG-Powered Knowledge System** | Auto-Switching: Gemini ↔ Qwen | Tiered Learning Pathways"
    )

    with gr.Tabs():
        with gr.TabItem("🔍 Ask & Compare"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧭 Learning Pathway")
                    level_selector = gr.Radio(
                        choices=["beginner", "intermediate", "advanced"],
                        value="beginner", label="Select your level"
                    )
                    faq_display = gr.Markdown(value=get_faq_list("beginner"))

                with gr.Column(scale=2):
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your question",
                            placeholder="e.g., What is the difference between oil, natural gas, and coal?",
                            lines=1, scale=4
                        )
                        level_filter = gr.Dropdown(
                            choices=["All Levels", "🟢 Beginner", "🟡 Intermediate", "🔵 Advanced"],
                            value="All Levels", label="Filter", scale=1
                        )
                    submit_btn = gr.Button("🔍 Ask the Hub", variant="primary")
                    jumpstart_out = gr.Markdown(label="Jump-Start Answer")
                    with gr.Row():
                        rag_out = gr.Markdown(label="RAG Deep-Dive")
                        llm_out = gr.Markdown(label="LLM General Knowledge")

            level_selector.change(fn=get_faq_list, inputs=level_selector, outputs=faq_display)
            submit_btn.click(fn=query_hub, inputs=[query_input, level_filter], outputs=[jumpstart_out, rag_out, llm_out])
            query_input.submit(fn=query_hub, inputs=[query_input, level_filter], outputs=[jumpstart_out, rag_out, llm_out])

        with gr.TabItem("📚 Knowledge Base"):
            kb_display = gr.Markdown(value="⏳ Loading knowledge base...")
            refresh_btn = gr.Button("🔄 Refresh Status")
            refresh_btn.click(fn=get_ingestion_summary, inputs=None, outputs=kb_display)

        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                "## Energy Learning Hub\n\n"
                "This platform helps newcomers learn about the oil, gas, and energy industry "
                "through a RAG-powered system grounded in authoritative PDF sources.\n\n"
                "**Current data source:** Oil 101 by Morgan Downey\n\n"
                "**LLMs:** Gemini 2.5 Flash (primary) + Qwen 2.5-7B (fallback)\n\n"
                "**Built by:** Derek Eng | 2026"
            )


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. LAUNCH — Start server FIRST, then load pipeline         ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    # Start the heavy initialization in a background thread
    init_thread = threading.Thread(target=initialize_pipeline, daemon=True)
    init_thread.start()
    print(f"🚀 Starting Gradio server on 0.0.0.0:{PORT} (pipeline loading in background)...")

    # Launch Gradio IMMEDIATELY so Render detects the port
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        theme=gr.themes.Soft()
    )
