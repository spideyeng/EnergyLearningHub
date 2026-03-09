"""
Energy Learning Hub — RAG-Powered Knowledge System
Deployed on Render as a Gradio web service.
"""

import os
import glob
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from operator import itemgetter

# ==================================================================
# 1. CONFIGURATION
# ==================================================================
PDF_DIR = os.environ.get("PDF_DIR", "./data")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
PORT = int(os.environ.get("PORT", 7860))

# ==================================================================
# 2. LOAD & CHUNK PDFs
# ==================================================================
print("📖 Loading PDFs from:", PDF_DIR)
pdf_files = glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True)

all_docs = []
for pdf_path in sorted(pdf_files):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = os.path.basename(pdf_path)
        all_docs.extend(pages)
        print(f"  ✅ {os.path.basename(pdf_path)}: {len(pages)} pages")
    except Exception as e:
        print(f"  ❌ {os.path.basename(pdf_path)}: {e}")

print(f"📄 Total pages loaded: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"🧩 Total chunks: {len(chunks)}")

# ==================================================================
# 3. BUILD VECTOR STORE
# ==================================================================
print("🔧 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("📦 Building vector store...")
vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="energy_hub",
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
print(f"✅ Vector store: {vector_store._collection.count()} vectors")

# ==================================================================
# 4. CONFIGURE LLMs (Gemini + Qwen fallback)
# ==================================================================
gemini_key = os.environ.get("GEMINI_API_KEY", "")
hf_token = os.environ.get("HF_TOKEN", "")

# Primary: Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=gemini_key
)

# Fallback: Qwen via HuggingFace
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
qwen_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.1
)
qwen_llm = ChatHuggingFace(llm=qwen_endpoint)

# Resilient chain
llm_with_fallback = gemini_llm.with_fallbacks([qwen_llm])
print("✅ LLMs configured (Gemini → Qwen fallback)")

# ==================================================================
# 5. RAG PIPELINE
# ==================================================================
template = """You are an expert energy educator for a learning hub that teaches
newcomers about oil, gas, and the broader energy sector. Answer the question
using ONLY the provided context from the knowledge base.

Guidelines:
- Use clear, accessible language appropriate for the learner's level
- Include specific data, examples, or case studies from the context when available
- If the context doesn't contain enough information, say so honestly
- End with a suggestion for what to explore next

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

qa = (
    {
        "context": itemgetter("query") | retriever,
        "question": itemgetter("query")
    }
    | prompt
    | llm_with_fallback
    | StrOutputParser()
)
print("✅ RAG pipeline ready")

# ==================================================================
# 6. FAQ KNOWLEDGE BASE
# ==================================================================
FAQ_DATABASE = {
    "beginner": {
        "label": "🟢 Beginner — Foundations",
        "description": "Start here if you're new to energy concepts.",
        "color": "#4CAF50",
        "categories": {
            "Energy Basics ⚡": [
                {
                    "q": "What is the difference between oil, natural gas, and coal?",
                    "jumpstart": "Oil is a liquid fossil fuel used mainly for transportation and petrochemicals. Natural gas is gaseous, often used for heating and electricity. Coal is solid, historically used for power generation and steelmaking. Each has different carbon intensity and energy density.",
                    "sources": ["Introductory energy textbooks", "IEA/EIA primers"]
                },
                {
                    "q": "How is energy measured (joules, BTUs, kWh)?",
                    "jumpstart": "Energy is measured in joules (J), British Thermal Units (BTUs), or kilowatt-hours (kWh). For example, 1 kWh powers a 100-watt light bulb for 10 hours.",
                    "sources": ["General science references", "EIA energy glossary"]
                },
                {
                    "q": "What are primary vs. secondary energy sources?",
                    "jumpstart": "Primary sources (coal, crude oil, sunlight) exist in nature. Secondary sources (electricity, gasoline) are converted from primary sources for practical use.",
                    "sources": ["Introductory energy textbooks"]
                }
            ],
            "Energy Systems 🔌": [
                {
                    "q": "How is electricity generated from fossil fuels?",
                    "jumpstart": "Fossil fuels are burned to heat water, producing steam that spins turbines connected to generators. This mechanical energy is converted into electricity.",
                    "sources": ["Power systems textbooks", "EIA resources"]
                },
                {
                    "q": "What are upstream, midstream, and downstream operations?",
                    "jumpstart": "Upstream covers exploration and production. Midstream involves transportation and storage (pipelines, shipping). Downstream refers to refining and distribution into usable products like gasoline.",
                    "sources": ["Industry primers", "SPE resources"]
                },
                {
                    "q": "What are renewable energy sources?",
                    "jumpstart": "Renewables include solar, wind, hydro, and geothermal. They rely on naturally replenished resources and generally produce lower greenhouse gas emissions.",
                    "sources": ["IRENA reports", "Introductory energy textbooks"]
                }
            ],
            "Careers & Skills 🎓": [
                {
                    "q": "What skills are needed to work in the energy industry?",
                    "jumpstart": "Technical skills (engineering, geology, data analysis), business skills (economics, project management), and increasingly sustainability knowledge are valuable.",
                    "sources": ["SPE career guides", "AAPG resources"]
                }
            ]
        },
        "next_steps": {
            "Want technical detail?": "intermediate",
            "Want policy & global view?": "advanced"
        }
    },
    "intermediate": {
        "label": "🟡 Intermediate — Industry Insights",
        "description": "For learners who understand the basics and want to explore operations and markets.",
        "color": "#FF9800",
        "categories": {
            "Resources & Operations 🛢️": [
                {
                    "q": "What are proven reserves and how are they estimated?",
                    "jumpstart": "Proven reserves are quantities of oil or gas that geological and engineering data show can be recovered under current economic and operating conditions.",
                    "sources": ["Petroleum geology textbooks", "SPE Reserves Definitions"]
                },
                {
                    "q": "What is the difference between baseload and peak load power?",
                    "jumpstart": "Baseload power is the steady supply needed 24/7 (often from coal, nuclear, or hydro). Peak load power meets demand spikes (often from gas turbines or renewables with storage).",
                    "sources": ["Electricity markets textbooks", "Grid operations guides"]
                }
            ],
            "Markets & Economics 📈": [
                {
                    "q": "How are oil prices determined?",
                    "jumpstart": "Prices are influenced by supply and demand, geopolitical events, OPEC decisions, production costs, and futures trading. They can be highly volatile.",
                    "sources": ["BP Statistical Review", "Market analysis reports"]
                },
                {
                    "q": "What is the role of OPEC in global energy markets?",
                    "jumpstart": "OPEC coordinates oil production policies among member states to influence global supply and stabilize prices.",
                    "sources": ["OPEC annual reports", "Energy outlook publications"]
                }
            ],
            "Sustainability & Security 🌱": [
                {
                    "q": "What is carbon capture and storage (CCS)?",
                    "jumpstart": "CCS captures CO₂ emissions from power plants or industrial sites and stores them underground to reduce atmospheric emissions.",
                    "sources": ["CCS textbooks", "IEA CCS reports"]
                },
                {
                    "q": "What is energy security and why is it important?",
                    "jumpstart": "Energy security means ensuring reliable, affordable access to energy through diversifying sources, securing supply chains, and reducing import dependence.",
                    "sources": ["IEA energy security publications", "Policy reports"]
                },
                {
                    "q": "Which professional certifications are useful?",
                    "jumpstart": "Certifications like PMP (Project Management), CFA (energy finance), or specialized petroleum engineering courses can boost career prospects.",
                    "sources": ["Professional certification syllabi", "SPE Handbook"]
                }
            ]
        },
        "next_steps": {
            "Want market & geopolitics?": "advanced",
            "Want technical engineering?": "advanced"
        }
    },
    "advanced": {
        "label": "🔵 Advanced — Strategic & Global Perspectives",
        "description": "For learners ready to engage with complex policy, geopolitics, and future trends.",
        "color": "#2196F3",
        "categories": {
            "Policy & Regulation 🏛️": [
                {
                    "q": "How do international agreements (like the Paris Accord) shape energy policy?",
                    "jumpstart": "Agreements like the Paris Accord push countries to reduce emissions, invest in renewables, and set long-term climate targets that reshape national energy strategies.",
                    "sources": ["UNFCCC publications", "IEA policy reports"]
                },
                {
                    "q": "What are net-zero targets and how do they affect the energy transition?",
                    "jumpstart": "Net-zero means balancing CO₂ emissions with removals. These targets drive investment in renewables, CCS, and efficiency improvements across the energy sector.",
                    "sources": ["World Bank reports", "IEA Net Zero Roadmap"]
                }
            ],
            "Geopolitics & Strategy 🌐": [
                {
                    "q": "How do geopolitics influence oil and gas markets?",
                    "jumpstart": "Political tensions, sanctions, trade agreements, and regional conflicts can disrupt supply chains, shift trade flows, and cause price volatility in global energy markets.",
                    "sources": ["Energy geopolitics texts", "The Prize by Daniel Yergin"]
                },
                {
                    "q": "What is the impact of subsidies and regulation on energy markets?",
                    "jumpstart": "Subsidies can lower consumer costs but distort markets. Regulations set safety and environmental standards. Together, they shape investment decisions and energy mix.",
                    "sources": ["Energy economics journals", "World Bank policy papers"]
                }
            ],
            "Future of Energy 🚀": [
                {
                    "q": "How does the energy transition affect traditional oil and gas companies?",
                    "jumpstart": "Many companies are diversifying into renewables, hydrogen, and CCS. The transition creates both risks (stranded assets) and opportunities (new markets).",
                    "sources": ["Corporate sustainability reports", "IEA World Energy Outlook"]
                },
                {
                    "q": "How do grids and transmission systems work at scale?",
                    "jumpstart": "Power grids balance generation and demand in real-time across vast networks. Modern grids integrate renewables, storage, and smart technology for reliability.",
                    "sources": ["Power systems engineering texts", "Grid integration studies"]
                },
                {
                    "q": "What are the major career pathways in renewables vs. fossil fuels?",
                    "jumpstart": "Fossil fuel careers focus on geology, drilling, and refining. Renewable careers span solar/wind engineering, grid integration, policy, and energy storage. Many skills transfer between sectors.",
                    "sources": ["Workforce trend studies", "Professional association guides"]
                }
            ]
        },
        "next_steps": {}
    }
}

# ==================================================================
# 7. HELPER FUNCTIONS
# ==================================================================

def get_model_source(response_obj):
    """Identify whether Gemini or Qwen answered."""
    metadata = getattr(response_obj, "response_metadata", {})
    if "safety_ratings" in metadata or "prompt_feedback" in metadata:
        return "Gemini 2.5 Flash"
    return "Qwen 2.5 (HuggingFace Fallback)"


def find_jumpstart(question, level=None):
    """Search FAQ database for a matching jump-start answer."""
    q_lower = question.lower().strip()
    for lvl, data in FAQ_DATABASE.items():
        if level and lvl != level:
            continue
        for cat, faqs in data["categories"].items():
            for faq in faqs:
                if faq["q"].lower().strip() == q_lower:
                    return {
                        "answer": faq["jumpstart"],
                        "sources": faq["sources"],
                        "level": lvl,
                        "category": cat
                    }
    return None


def query_hub(user_query, selected_level):
    """Main query handler: jump-start + RAG + direct LLM."""
    level_map = {
        "All Levels": None,
        "🟢 Beginner": "beginner",
        "🟡 Intermediate": "intermediate",
        "🔵 Advanced": "advanced"
    }
    level_filter = level_map.get(selected_level)

    # Jump-start lookup
    jumpstart_result = find_jumpstart(user_query, level_filter)
    if jumpstart_result:
        lvl = jumpstart_result['level'].capitalize()
        cat = jumpstart_result['category']
        ans = jumpstart_result['answer']
        srcs = ', '.join(jumpstart_result['sources'])
        jumpstart_text = (
            "📋 **Jump-Start Answer** (" + lvl + " | " + cat + ")"
            + "\n\n" + ans
            + "\n\n📚 Suggested sources: " + srcs
        )
    else:
        jumpstart_text = "_No matching FAQ found — showing RAG retrieval only._"

    # RAG deep-dive
    try:
        rag_response = qa.invoke({"query": user_query})
        rag_text = "📚 **RAG Deep-Dive Answer**" + "\n\n" + str(rag_response)
    except Exception as e:
        rag_text = "❌ RAG Error: " + str(e)

    # Direct LLM comparison
    try:
        llm_response = llm_with_fallback.invoke([HumanMessage(content=user_query)])
        model_name = get_model_source(llm_response)
        llm_text = "🤖 [Source: " + model_name + "]" + "\n\n" + llm_response.content
    except Exception as e:
        llm_text = "❌ LLM Error: " + str(e)

    return jumpstart_text, rag_text, llm_text


def get_faq_list(level_key):
    """Return formatted FAQ list for a given level."""
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


# ==================================================================
# 8. GRADIO UI
# ==================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Energy Learning Hub") as demo:

    gr.Markdown(
        "# 🌍 Energy Learning Hub\n"
        "**RAG-Powered Knowledge System** | Auto-Switching: Gemini ↔ Qwen | Tiered Learning Pathways"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🧭 Learning Pathway")
            level_selector = gr.Radio(
                choices=["beginner", "intermediate", "advanced"],
                value="beginner",
                label="Select your level",
                info="Choose a tier to see guided FAQs"
            )
            faq_display = gr.Markdown(value=get_faq_list("beginner"))

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask a Question")
            gr.Markdown("_Click any FAQ from the left, or type your own question below._")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g., What is the difference between oil, natural gas, and coal?",
                    lines=1,
                    scale=4
                )
                level_filter = gr.Dropdown(
                    choices=["All Levels", "🟢 Beginner", "🟡 Intermediate", "🔵 Advanced"],
                    value="All Levels",
                    label="Filter",
                    scale=1
                )

            submit_btn = gr.Button("🔍 Ask the Hub", variant="primary")

            jumpstart_output = gr.Markdown(label="Jump-Start Answer")
            with gr.Row():
                rag_output = gr.Markdown(label="RAG Deep-Dive")
                llm_output = gr.Markdown(label="LLM General Knowledge")

    # Wire events
    level_selector.change(fn=get_faq_list, inputs=level_selector, outputs=faq_display)
    submit_btn.click(fn=query_hub, inputs=[query_input, level_filter], outputs=[jumpstart_output, rag_output, llm_output])
    query_input.submit(fn=query_hub, inputs=[query_input, level_filter], outputs=[jumpstart_output, rag_output, llm_output])


# ==================================================================
# 9. LAUNCH — Render requires 0.0.0.0 and uses PORT env var
# ==================================================================
if __name__ == "__main__":
    print(f"🚀 Launching on port {PORT}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False
    )
