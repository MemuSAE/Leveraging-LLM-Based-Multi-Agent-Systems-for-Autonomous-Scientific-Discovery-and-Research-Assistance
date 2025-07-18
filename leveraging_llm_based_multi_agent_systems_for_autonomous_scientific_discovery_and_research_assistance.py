# -*- coding: utf-8 -*-
"""Leveraging LLM-Based Multi-Agent Systems for Autonomous Scientific Discovery and Research Assistance.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SiSx6E8Tu8WMPrOrqs3UrQO97wNlDUCX

# Leveraging LLM-Based Multi-Agent Systems for Autonomous Scientific Discovery and Research Assistance

## Installing Dependeancies
"""

! pip install langchain
!pip install transformers accelerate bitsandbytes
! pip install sentence-transformers
!pip install langchain langchain-community
! pip install pypdf
!apt install libomp-dev
!pip install faiss-cpu
!pip install InstructorEmbedding

"""## Model setup"""

import os
import time
import torch
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)


# 0) Verify GPU

if not torch.cuda.is_available():
    raise EnvironmentError("No CUDA GPU detected.")
print("Using CUDA device:", torch.cuda.get_device_name(0))


# 1) Prompt templates

PROPOSER_PROMPT = (
    "You are an interdisciplinary researcher. Based on the following summarized literature:\n\n"
    "{context}\n\n"
    "Propose 2 novel, plausible hypotheses. List each clearly."
)
VALIDATOR_PROMPT = (
    "You are a rigorous scientific validator. Given the hypothesis:\n\n"
    "\"{hypothesis}\"\n\n"
    "And this summarized context:\n\n"
    "{context}\n\n"
    "1) Rate feasibility 1–10.\n"
    "2) Summarize supporting/contradicting evidence.\n"
    "3) Note assumptions or missing data.\n"
)
GAP_PROMPT = (
    "Analyze this summarized context:\n\n{context}\n\n"
    "Identify 2–3 high-priority research gaps, with brief justification."
)


# 2) MultiAgentPipeline with summarization

class MultiAgentPipeline:
    def __init__(
        self,
        pdf_paths: List[str],
        faiss_path: str = "faiss_index",
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        summarizer_model: str = "facebook/bart-large-cnn",
        llm_model: str = "Qwen/Qwen3-1.7B",
    ):
        self.pdf_paths = pdf_paths
        self.faiss_path = faiss_path


        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")
        if not self.hf_token:
            raise EnvironmentError("Set HUGGINGFACEHUB_API_TOKEN")

        # 1) Build & load FAISS
        self._load_or_build_faiss(emb_model)

        # 2) Summarizer pipeline (GPU)
        self.summarizer = hf_pipeline(
            "summarization",
            model=summarizer_model,
            device=0,
            tokenizer=summarizer_model,
        )

        # 3) LLM pipeline (fp16, GPU)
        self._init_llm(llm_model)

        # 4) Thread pool
        self.executor = ThreadPoolExecutor(max_workers=3)

    def _load_or_build_faiss(self, emb_model: str):
        embedder = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs={"device": "cuda", "use_auth_token": self.hf_token},
        )

        if os.path.isdir(self.faiss_path):
            self.vectorstore = FAISS.load_local(
                self.faiss_path, embedder, allow_dangerous_deserialization=True
            )
            return

        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for path in self.pdf_paths:
            pages = PyPDFLoader(path).load()
            docs.extend(splitter.split_documents(pages))

        texts = [doc.page_content for doc in docs]
        embeddings: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings.extend(embedder.embed_documents(batch))

        pairs: List[Tuple[str, List[float]]] = list(zip(texts, embeddings))
        self.vectorstore = FAISS.from_embeddings(pairs, embedder)
        self.vectorstore.save_local(self.faiss_path)

    def _init_llm(self, llm_model: str):
        tokenizer = AutoTokenizer.from_pretrained(llm_model, use_auth_token=self.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=self.hf_token,
        )

        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        text_gen = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=text_gen)

    def _get_context(self, k: int = 3) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents("")


        futures = [
            self.executor.submit(
                self.summarizer,
                d.page_content,
                max_length=150,
                min_length=30,
                do_sample=False,
            )
            for d in docs
        ]
        summaries = [f.result()[0]["summary_text"] for f in futures]
        return "\n\n".join(summaries)

    def _propose(self, context: str) -> List[str]:
        out = self.llm.generate([PROPOSER_PROMPT.format(context=context)])
        text = out.generations[0][0].text
        return [h.strip() for h in text.split("\n") if h.strip()]

    def _validate(self, hypos: List[str], context: str) -> List[str]:
        def task(hypo: str) -> str:
            out = self.llm.generate(
                [VALIDATOR_PROMPT.format(hypothesis=hypo, context=context)]
            )
            return out.generations[0][0].text

        return list(self.executor.map(task, hypos))

    def _analyze_gaps(self, context: str) -> str:
        out = self.llm.generate([GAP_PROMPT.format(context=context)])
        return out.generations[0][0].text

    def run(self) -> Dict[str, Any]:
        start = time.time()

        t0 = time.time()
        context = self._get_context(k=3)
        print(f"[{time.time()-t0:.1f}s] Retrieved + summarized context")

        t1 = time.time()
        hypotheses = self._propose(context)
        print(f"[{time.time()-t1:.1f}s] Proposed {len(hypotheses)} hypotheses")

        t2 = time.time()
        validations = self._validate(hypotheses, context)
        print(f"[{time.time()-t2:.1f}s] Validations done")

        t3 = time.time()
        gaps = self._analyze_gaps(context)
        print(f"[{time.time()-t3:.1f}s] Gap analysis done")

        print(f" Total time: {time.time()-start:.1f}s\n")
        return {
            "proposed_hypotheses": hypotheses,
            "validations": validations,
            "research_gaps": gaps,
        }



# Example usage

if __name__ == "__main__":
    pdfs = ["/content/2003.01332v1.pdf", "/content/2303.06455v2.pdf"]
    pipe = MultiAgentPipeline(pdfs)
    results = pipe.run()

    print("--- Hypotheses ---")
    for h in results["proposed_hypotheses"]:
        print("-", h)

    print("\n--- Validations ---")
    for v in results["validations"]:
        print(v, "\n")

    print("\n--- Research Gaps ---")
    print(results["research_gaps"])

"""## Evaluation"""

import os
import numpy as np
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class MultiAgentEvaluator:
    """
    Evaluates hypotheses (and, if desired, gap analyses) produced by MultiAgentPipeline by measuring
    their embedding‐similarity to the indexed context chunks.
    """

    def __init__(
        self,
        faiss_path: str = "faiss_index",
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_token: str = "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",
        hf_token_env: str = "HUGGINGFACEHUB_API_TOKEN",
    ):

        token = hf_token or os.getenv(hf_token_env, "")
        if not token:
            raise EnvironmentError(
                f"Set ${hf_token_env} or pass hf_token directly to load embeddings"
            )

        self.embedder = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs={"device": "cuda", "use_auth_token": token},
        )

        if not os.path.isdir(faiss_path):
            raise FileNotFoundError(f"FAISS index directory '{faiss_path}' not found.")
        self.vectorstore = FAISS.load_local(
            faiss_path, self.embedder, allow_dangerous_deserialization=True
        )

    def evaluate_hypotheses(
        self,
        hypotheses: List[str],
        k: int = 3,
        support_threshold: float = 0.5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each hypothesis:
          1. Retrieve top‐k context chunks (via embedding‐similarity).
          2. Compute average similarity.
          3. Flag as “supported” if avg ≥ support_threshold.

        Returns a dict mapping each hypothesis to its avg_similarity, top_k_scores, and supported flag.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for hypo in hypotheses:

            docs_and_scores = self.vectorstore.similarity_search_with_score(hypo, k=k)
            scores = [float(score) for (_, score) in docs_and_scores]
            avg_sim = float(np.mean(scores)) if scores else 0.0
            results[hypo] = {
                "avg_similarity": avg_sim,
                "top_k_scores": scores,
                "supported": avg_sim >= support_threshold,
            }
        return results

    def evaluate_gap_analysis(
        self,
        gap_text: str,
        k: int = 3,
        grounding_threshold: float = 0.3,
    ) -> float:
        """
        Measures how “novel” the gap analysis is (i.e. low grounding means it truly
        identifies new topics). Returns the average similarity to context (lower is better).
        """
        docs_and_scores = self.vectorstore.similarity_search_with_score(gap_text, k=k)
        scores = [float(score) for (_, score) in docs_and_scores]
        return float(np.mean(scores)) if scores else 0.0

if __name__ == "__main__":
    pdfs = ["/content/2003.01332v1.pdf", "/content/2303.06455v2.pdf"]
    pipe = MultiAgentPipeline(pdfs)
    results = pipe.run()

    evaluator = MultiAgentEvaluator(faiss_path=pipe.faiss_path)
    hypo_metrics = evaluator.evaluate_hypotheses(
        results["proposed_hypotheses"],
        k=3,
        support_threshold=0.5
    )

    print("\n--- Hypothesis Evaluation ---")
    for h, m in hypo_metrics.items():
        print(f"Hypothesis: {h}")
        print(f"  Avg Similarity: {m['avg_similarity']:.3f}")
        print(f"  Top-{len(m['top_k_scores'])} scores: {m['top_k_scores']}")
        print(f"  Supported by context? {'Yes' if m['supported'] else 'No'}\n")


    gap_grounding = evaluator.evaluate_gap_analysis(
        results["research_gaps"], k=3
    )
    print(f"Gap Analysis avg similarity to context: {gap_grounding:.3f}")

import re
import csv
import time
import statistics
from typing import List, Dict, Any

# from your_module import MultiAgentPipeline  # ← adjust to your import path


def parse_feasibility_scores(validations: List[str]) -> List[int]:
    """
    Extract the integer rating (1–10) from each validator's output.
    Tries to match the number right after the "1)" answer marker.
    Falls back to any standalone 1–10 integer if needed.
    """
    scores: List[int] = []
    for text in validations:
        # 1) look for a line starting with "1)" then digits
        m = re.search(
            r'^\s*1\)\s*(?:Rate feasibility[:\-\s]*)?(\b(?:[1-9]|10)\b)',
            text,
            re.MULTILINE,
        )
        if not m:
            # 2) fallback: first standalone integer between 1 and 10
            m = re.search(r'\b([1-9]|10)\b', text)
        scores.append(int(m.group(1)) if m else None)
    return scores


def run_experiment(
    name: str,
    pdf_paths: List[str],
    retriever_k: int = 3,
    max_hypotheses: int = 2,
) -> Dict[str, Any]:
    """
    Runs the pipeline once, enforces exactly `max_hypotheses`, and returns:
      - runtime
      - sliced hypotheses & validations
      - parsed feasibility scores
      - raw gap analysis
    """
    print(f"\n Experiment: {name}")
    start = time.time()

    # init pipeline (will build/load FAISS)
    pipe = MultiAgentPipeline(pdf_paths, faiss_path=f"faiss_{name}")
    # patch k for retrieval
    pipe._get_context = lambda k=retriever_k: MultiAgentPipeline._get_context(pipe, k=k)

    # run end-to-end
    out = pipe.run()

    # enforce only the first `max_hypotheses`
    hypos = out["proposed_hypotheses"][:max_hypotheses]
    vals = out["validations"][: len(hypos)]
    runtime = time.time() - start

    scores = parse_feasibility_scores(vals)
    return {
        "name": name,
        "runtime": runtime,
        "hypotheses": hypos,
        "validations": vals,
        "scores": scores,
        "gaps": out["research_gaps"],
    }


def summarize_scores(scores: List[int]) -> Dict[str, float]:
    """Compute min, max, mean, std (ignoring None)."""
    clean = [s for s in scores if s is not None]
    if not clean:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


if __name__ == "__main__":

    EXPERIMENTS = [
        {
            "name": "default_k3",
            "pdfs": ["/content/2003.01332v1.pdf", "/content/2303.06455v2.pdf"],
            "k": 3,
        },
        {
            "name": "larger_k5",
            "pdfs": ["/content/2003.01332v1.pdf", "/content/2303.06455v2.pdf"],
            "k": 5,
        },

    ]


    rows = []
    for exp in EXPERIMENTS:
        res = run_experiment(
            name=exp["name"],
            pdf_paths=exp["pdfs"],
            retriever_k=exp["k"],
            max_hypotheses=2,
        )
        stats = summarize_scores(res["scores"])
        rows.append(
            {
                "experiment": res["name"],
                "runtime_s": round(res["runtime"], 2),
                "num_hypotheses": len(res["hypotheses"]),
                "min_score": stats["min"],
                "max_score": stats["max"],
                "mean_score": round(stats["mean"], 2) if stats["mean"] is not None else "",
                "std_score": round(stats["std"], 2) if stats["std"] is not None else "",
            }
        )


    fieldnames = [
        "experiment",
        "runtime_s",
        "num_hypotheses",
        "min_score",
        "max_score",
        "mean_score",
        "std_score",
    ]
    with open("evaluation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n Done! Results saved to evaluation_results.csv")
    for row in rows:
        print(row)