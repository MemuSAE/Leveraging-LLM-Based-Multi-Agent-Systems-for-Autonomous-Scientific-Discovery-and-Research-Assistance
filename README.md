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

1. Place your scientific PDFs in a directory (e.g., `./data/`).

2. Run the main script:

   ```bash
   python main.py
   ```

   This will:

   - Load and process PDFs.
   - Build/load a FAISS index.
   - Generate and validate hypotheses.
   - Identify research gaps.
   - Evaluate outputs and save results to `evaluation_results.csv`.

3. Example configuration in `main.py`:

   ```python
   pdfs = ["./data/2003.01332v1.pdf", "./data/2303.06455v2.pdf"]
   pipe = MultiAgentPipeline(pdfs)
   results = pipe.run()
   ```

4. Experiment with different settings: Modify the `EXPERIMENTS` list in `main.py` to adjust parameters like `retriever_k` or PDF inputs.

## Files

- `main.py`: Core script for running the multi-agent pipeline and experiments.
- `requirements.txt`: List of required Python packages.
- `evaluation_results.csv`: Output file with experiment results (runtime, hypothesis counts, feasibility scores).
- `faiss_index/`: Directory storing the FAISS vector store (auto-generated).

## Requirements

See `requirements.txt` for a full list. Key dependencies include:

- `langchain`
- `transformers`
- `sentence-transformers`
- `pypdf`
- `faiss-cpu`
- `InstructorEmbedding`
- `torch`

## Output

The system produces:

- **Hypotheses**: 2 novel hypotheses based on the input literature.
- **Validations**: Feasibility scores (1–10), evidence summaries, and assumptions for each hypothesis.
- **Research Gaps**: 2–3 identified gaps with justifications.
- **Evaluation Metrics**: Embedding-based similarity scores for hypotheses and gap analysis, saved to `evaluation_results.csv`.

## Example Output

```plaintext
--- Hypotheses ---
- Hypothesis 1: [Generated hypothesis text]
- Hypothesis 2: [Generated hypothesis text]

--- Validations ---
Hypothesis: [Hypothesis 1]
  Feasibility: 7/10
  Evidence: [Summary of supporting/contradicting evidence]
  Assumptions: [Noted assumptions or missing data]

--- Research Gaps ---
- Gap 1: [Description and justification]
- Gap 2: [Description and justification]

--- Evaluation ---
Hypothesis: [Hypothesis 1]
  Avg Similarity: 0.612
  Top-3 scores: [0.65, 0.60, 0.58]
  Supported by context? Yes
Gap Analysis avg similarity to context: 0.287
```

## Notes

- Ensure a CUDA-compatible GPU is available, as the system relies on GPU acceleration.
- The default LLM (`Qwen/Qwen3-1.7B`) and embedding model (`all-MiniLM-L6-v2`) can be swapped for other Hugging Face models.
- FAISS index persistence allows reusing embeddings across runs.
- The system assumes at least one valid PDF input and a Hugging Face API token.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
