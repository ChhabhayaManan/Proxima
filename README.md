# Proxima: AI Model Benchmarking Simulator

Proxima is an enterprise-grade framework designed to evaluate and benchmark how well different Large Language Models (LLMs) perform code reviews. Inspired by the checklist-based evaluation methodologies outlined in the Sphinx research architecture, Proxima systematically assesses an AI's ability to catch real-world bugs in Pull Requests.

> **Note:** While Proxima includes a streamlined web interface, the core focus of this project is its **robust benchmarking engine and multi-agent workflow pipeline**.

---

## 📸 UI Dashboard

*(![Proxima Streamlit UI Screenshot](./assets/ui_screenshot.png))*  
*Space for Streamlit UI Screenshot demonstrating the workflow execution and evaluation results.*

---

## 🚀 The Benchmarking Methodology

Evaluating an LLM's code review capabilities is challenging because "good code" can be subjective. Proxima solves this by relying on historical ground truth:

1. **The Ground Truth (Data Generation)**: We reverse-engineer a historically merged Pull Request to identify the *actual* bugs and modifications the human developer fixed. These form a definitive "Review Checklist."
2. **The Test (Model Under Evaluation)**: We present the original, broken code to your selected AI Model and ask it to conduct a code review.
3. **The Score**: We mathematically calculate the intersection between the AI's findings and the ground-truth checklist, producing a definitive coverage score representing what percentage of real-world bugs the AI successfully caught natively.

---

## ⚙️ Architecture & Agentic Workflow

Proxima is powered by **LangGraph** and **Langchain**, utilizing a sequential multi-agent workflow that is separated into two primary engines:

### 1. Data Generation Engine (Workflow 1)
This phase is responsible for creating the ground truth dataset from real GitHub PRs.
* **`gitPRRetriever`**: Fetches PR metadata, diffs, and codebase context directly via the GitHub API.
* **`prReviewInstructionAgent` / `pseudoSolutionAgent`**: Analyzes the original and merged code to generate pseudo-modified code and structured review instructions.
* **`checklistGenerationAgent`**: Converts the structured review into a definitive, granular ground-truth **Checklist**.

### 2. Checklist Evaluation Engine (Workflow 2)
This phase evaluates the target LLM against the generated checklist.
* **`llmGeneratedPRReviewAgent`**: The target LLM acts as the reviewer and generates a PR review based solely on the pre-merge codebase and diffs.
* **`scoreGenerationAgent`**: Compares the AI's generated review against the ground-truth Checklist, evaluating coverage and computing the final structured evaluation score.

---

## 🛠️ Technology Stack

* **Workflow Orchestration**: LangGraph, LangChain
* **LLM Providers**: Google Gemini, Groq, Ollama (Local/Cloud Inference)
* **Frontend UI**: Streamlit
* **Code Analysis**: GitHub REST API, AST/Diff Analysis
* **Core Language**: Python

---

## 🚦 Getting Started

### Prerequisites
* Python 3.9+
* GitHub Personal Access Token (`GITHUB_TOKEN`)
* API Keys for your preferred LLM providers (Gemini, Groq) or a running Ollama instance.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Proxima
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Copy the example environment file and add your credentials:
   ```bash
   cp .env.example .env
   ```
   *Make sure to populate your `GITHUB_TOKEN` and any provider API keys.*

### Running the Simulator

Launch the Streamlit dashboard to configure your models and run the benchmarking pipeline:

```bash
streamlit run app.py
```

1. **Select Data Generation Model**: Choose the model responsible for creating the rigorous ground-truth checklist.
2. **Select Evaluation Provider**: Choose the LLM you want to benchmark (e.g., a local Ollama model or Groq).
3. **Target a Repository**: Enter the repository owner, name, and select a merged PR to use as the test case.
4. **Run Workflow**: Watch as the multi-agent pipeline extracts data, formulates the test, queries the target model, and calculates the final coverage score.
