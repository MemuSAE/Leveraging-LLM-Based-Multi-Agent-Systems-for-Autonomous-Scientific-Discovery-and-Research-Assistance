# LLM-Based Multi-Agent System for Autonomous Scientific Discovery

## Overview
This repository contains a Python implementation of a multi-agent system leveraging large language models (LLMs) for autonomous scientific discovery and research assistance. The system processes scientific literature (PDFs), generates novel hypotheses, validates them, identifies research gaps, and evaluates outputs using embedding-based similarity metrics. It is designed to run on GPU-enabled environments, utilizing frameworks like LangChain, Transformers, and FAISS.

## Features
- **PDF Processing**: Extracts and chunks text from scientific PDFs using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Context Retrieval**: Builds a FAISS vector store with embeddings from `sentence-transformers/all-MiniLM-L6-v2` for efficient context retrieval.
- **Hypothesis Generation**: Proposes novel hypotheses based on summarized literature using a customizable LLM (default: `Qwen/Qwen3-1.7B`).
- **Validation**: Validates hypotheses by assessing feasibility, supporting/contradicting evidence, and identifying assumptions.
- **Gap Analysis**: Identifies high-priority research gaps with justifications.
- **Evaluation**: Measures hypothesis support and gap novelty using embedding similarity against the literature context.
- **Experimentation**: Runs controlled experiments with configurable retrieval parameters and logs results to CSV.

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU
- Required packages (see `requirements.txt`)
- Hugging Face API token (`HUGGINGFACEHUB_API_TOKEN`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-multi-agent-discovery.git
   cd llm-multi-agent-discovery
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install `libomp-dev` (for FAISS):
   ```bash
   apt install libomp-dev
   ```
4. Set your Hugging Face API token:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN='your-token-here'
   ```

## Usage
1. Place
