# finetune-hub: Math-to-LaTeX VLM Pipeline

This repository contains a modular, easy-to-run pipeline for fine-tuning Vision-Language Models (VLMs) like **PaliGemma-3B** on handwritten math equations. It utilizes **QLoRA (4-bit quantization)** to enable training on free Google Colab GPUs (T4).

---

## Quick Start (Google Colab)

The easiest way to run this pipeline is using the provided Notebook.

### **1. Prerequisites**
Before running the code, you need two things:

1.  **Hugging Face Account:** You must have an account to access the base model (`google/paligemma-3b-pt-224`).
    * **Action:** Go to [Hugging Face - PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) and click the button **"Agree and Access Repository"** to accept the license.
2.  **Access Token:**
    * Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens).
    * Create a new token (Role: `Read`).
    * **Copy it.** You will need it in step 3.

### **2. Setup in Colab**
1.  **Upload Files:**
    * Upload the `finetune_hub` folder and `Fine Tuning.ipynb` to your Google Drive (e.g., inside a folder named `Colab Notebooks`).
2.  **Open the Notebook:**
    * Double-click `Fine Tuning.ipynb` to open it in Colab.
3.  **Set HF_TOKEN Secret (Crucial Step):**
    * In the Colab sidebar (on the left), click the **Key icon** (Secrets).
    * Add a new secret:
        * **Name:** `HF_TOKEN`
        * **Value:** *(Paste your Hugging Face token here)*
    * Toggle the "Notebook access" switch to **ON**.

---

## How to Run the Pipeline

The notebook is automated. Simply run the cells in order.