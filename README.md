# 🌍 Energy Learning Hub — Deploy to Render

A RAG-powered energy education platform with tiered FAQ pathways, Gemini/Qwen auto-fallback, and an interactive Gradio UI.

Presentation deck on this Energy Learning Hub [Energy_Learning_Hub_Deck.pptx](https://github.com/spideyeng/EnergyLearningHub/blob/7bc919632e4772c91fad32defac25805b04f4725/presentation/Energy_Learning_Hub_Deck.pptx)

---

## Prerequisites

- A **GitHub** account
- A **Render** account (free at [render.com](https://render.com))
- Your **GEMINI_API_KEY** (from [Google AI Studio](https://aistudio.google.com/apikey))
- Your **HF_TOKEN** (from [Hugging Face Settings](https://huggingface.co/settings/tokens))
- Your energy **PDF files** ready to upload

---

## Step-by-Step Deployment

### Step 1 — Push to GitHub

```bash
# Clone or create your repo
git init energy-learning-hub
cd energy-learning-hub

# Copy all project files into this folder:
#   app.py, requirements.txt, Dockerfile,
#   render.yaml, .gitignore, data/

# Add your PDF files to the data/ folder
cp ~/your-pdfs/*.pdf data/

# Push to GitHub
git add .
git commit -m "Initial commit — Energy Learning Hub"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/EnergyLearningHub.git
git push -u origin main
```

### Step 2 — Create the Web Service on Render

1. Go to [render.com/dashboard](https://dashboard.render.com)
2. Click **New** → **Web Service**
3. Select **Build and deploy from a Git repository** → click **Next**
4. Connect your GitHub account and select your `energylearninghub` repo
5. Configure the service:

| Setting          | Value                             |
|------------------|-----------------------------------|
| **Name**         | `energylearninghub`               |
| **Region**       | Singapore (closest to you)        |
| **Branch**       | `main`                            |
| **Runtime**      | Docker                            |
| **Instance Type**| Starter ($7/mo) or Free*          |

> *Free tier works but spins down after 15 minutes of inactivity, causing a cold-start delay of 2–3 minutes on the next visit. Starter tier stays always-on.

### Step 3 — Set Environment Variables

In the Render service settings, go to **Environment** and add:

| Key              | Value                                   |
|------------------|-----------------------------------------|
| `GEMINI_API_KEY` | Your Google AI API key                  |
| `HF_TOKEN`       | Your Hugging Face access token          |
| `PDF_DIR`        | `./data`                                |
| `CHROMA_DIR`     | `./chroma_db`                           |

### Step 4 — Deploy

Click **Create Web Service**. Render will:

1. Pull your code from GitHub
2. Build the Docker image (~5–10 min first time)
3. Install dependencies
4. Load your PDFs and build the vector store
5. Launch the Gradio app

Once complete, your app is live at:
```
https://energylearninghub.onrender.com
```

---

## Project Structure

```
energylearninghub/
├── app.py              ← Main application (RAG + Gradio UI)
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Container build instructions
├── render.yaml         ← Render Blueprint (one-click deploy)
├── .gitignore
└── data/
    ├── README.md
    ├── energy-textbook.pdf      ← Your PDFs go here
    ├── iea-report-2024.pdf
    └── ...
```

---

## One-Click Deploy (Alternative)

If you've pushed to GitHub, you can also use the Render Blueprint:

1. Go to [render.com/dashboard](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Connect your repo (which contains `render.yaml`)
4. Render reads the blueprint and pre-fills all settings
5. Enter your `GEMINI_API_KEY` and `HF_TOKEN` when prompted
6. Click **Deploy**

---

## Adding or Updating PDFs

To add new PDF documents after deployment:

1. Add the PDFs to the `data/` folder in your repo
2. Commit and push:
   ```bash
   git add data/
   git commit -m "Add new energy PDFs"
   git push
   ```
3. Render auto-redeploys on every push to `main`

> The vector store is rebuilt on each deploy. For large PDF collections, consider adding a persistent disk (configured in `render.yaml`) to cache the ChromaDB between deploys.

---

## Troubleshooting

**Build fails with memory error**
→ The free tier has 512 MB RAM. Large PDF collections or the embedding model may exceed this. Upgrade to Starter ($7/mo, 2 GB RAM).

**App takes long to start**
→ First deploy downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and indexes all PDFs. Subsequent deploys are faster if using a persistent disk.

**"Both models failed" error**
→ Check that `GEMINI_API_KEY` and `HF_TOKEN` are correctly set in Render's Environment tab. Gemini has a daily request limit on the free tier.

**Port binding error**
→ The app reads the `PORT` environment variable (Render sets this to `10000`). Don't hardcode a different port.

---

## Local Development

To test locally before deploying:

```bash
# Set environment variables
export GEMINI_API_KEY="your-key"
export HF_TOKEN="your-token"
export PDF_DIR="./data"

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

The app opens at `http://localhost:10000`.
