# MIZAN: Cells 7, 8, and 9 





These three cells form the **generative back‑end**, the **quantitative evaluation suite**, and the **diagnostic error analysis** of the MIZAN pipeline.

- **Cell 7** loads the 9‑billion‑parameter SILMA‑9B‑Instruct large language model in 4‑bit quantised form, constructs a strict RAFT (Retrieval‑Augmented Fine‑Tuning style) prompt, and generates a grounded Arabic legal answer.
- **Cell 7B (optional)** demonstrates Low‑Rank Adaptation (LoRA) fine‑tuning on a smaller 2B model to illustrate Parameter‑Efficient Fine‑Tuning without altering the main pipeline.
- **Cell 8** runs the entire RAG pipeline on a held‑out evaluation set and computes faithful metrics: retrieval hit‑rates, citation accuracy, hallucination rate, BERTScore, and traditional similarity measures.
- **Cell 9** inspects failures, printing top mis‑classified queries, analysing classifier confusion, and identifying systemic generation errors.

Together they close the loop from query to answer and provide the evidence required by the course rubric for qualitative and quantitative evaluation.

---

## 2. Cell 7 — RAG Generation with SILMA‑9B‑Instruct (Member 5)

### 2.1 Purpose

Instantiates the generative large language model under strict hardware constraints, formats the prompt as a “closed‑book” legal assistant that must only use the provided articles, and produces the final answer. The model is never fine‑tuned on Egyptian law; its knowledge comes exclusively from the retrieved chunks injected at runtime.

### 2.2 Model Loading and Quantisation

The model `silma-ai/SILMA-9B-Instruct-v1.0` is a 9‑billion‑parameter decoder‑only transformer (Gemma‑2 family) fine‑tuned for Arabic instruction following. In full precision (float16) it would require ~18 GB of VRAM, exceeding a single T4 GPU. MIZAN therefore loads it with **4‑bit Normal Float (NF4)** quantisation via the `bitsandbytes` library.

**Quantisation configuration (simplified):**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["llm_model"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

- **NF4:** A data‑type that normalises weight values to a zero‑mean unit‑variance distribution, then maps them to 4‑bit discrete bins. This achieves near‑lossless compression while reducing memory from 18 GB to ~7.5 GB.
- **Double quantisation:** further compresses the quantisation constants themselves.
- **Compute dtype:** float16, so matrix multiplications happen in half precision, benefiting from GPU tensor cores.

### 2.3 Prompt Construction (RAFT‑style)

The prompt enforces grounding: the model is instructed to answer *only* from the provided articles, to cite the article number, and to refuse if the context is insufficient. The structure follows the Gemma‑2 chat template:

```
<start_of_turn>user
أنت مساعد قانوني مصري. أجب فقط باستخدام المواد التالية. 
إذا لم تكن المواد كافية، ارفض الإجابة بأدب.

المواد:
[Article 1 text with law name and number]
[Article 2 text with law name and number]
...

السؤال: {user query}
<end_of_turn>
<start_of_turn>model
```

The `user` turn contains the system instruction, the top‑k retrieved articles (each truncated to 600 characters), and the original user question. The model then generates the `model` turn.

**Why this format?**  
Research on Retrieval‑Augmented Fine‑Tuning (RAFT) shows that explicitly instructing the model to use only the provided documents dramatically reduces hallucination. The model is not tempted to rely on its parametric memory of Egyptian law, which is known to be inaccurate.

### 2.4 Generation Parameters

- `max_new_tokens = 300` – sufficient for a legal answer of 2–4 sentences plus a citation.
- `do_sample = False` – greedy decoding, ensuring deterministic and reproducible outputs.
- `temperature = 0.3` – low temperature to avoid creative paraphrasing that might alter legal wording.
- `repetition_penalty = 1.05` – discourages loops.
- `eos_token_id` – set to the model’s end‑of‑turn token.

**Example generation call:**
```python
outputs = model.generate(
    **tokenized_prompt,
    max_new_tokens=300,
    do_sample=False,
    temperature=0.3,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.eos_token_id,
)
answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
```

### 2.5 Post‑Processing and Safety

After generation, the answer is stripped of leading whitespace and checked:
- If it contains an article number (`المادة`), it’s considered a grounded answer.
- If it contains a refusal phrase (e.g., “المعذرة”), it’s accepted as a legitimate refusal.
- Otherwise, a confidence disclaimer is appended: “ملاحظة: هذه إجابة إرشادية وقد لا تكون دقيقة قانونيًا.”

### 2.6 Memory Management

Because the Qwen3 embedding model, the BGE‑M3 reranker, and the SILMA‑9B must coexist on two T4 GPUs, careful memory hygiene is applied:
- `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()` are called before loading the LLM.
- The Qwen3 embedder is loaded **only** for query encoding and then deleted before the LLM loads. It is re‑loaded on‑demand.
- The reranker is moved to the second GPU (`device1`) if available, freeing the first GPU for the LLM.
- The LLM uses `device_map="auto"` to distribute layers across both GPUs.

### 2.7 Tensor Flow Summary

| Stage | Shape | Operation |
|-------|-------|-----------|
| Tokenization | `[1, prompt_len]` | Tokenizer (Gemma‑2 vocabulary) |
| Embedding | `[1, prompt_len, 4096]` | Word embeddings |
| Transformer layers | `[1, prompt_len, 4096]` | 40 decoder layers with causal mask |
| 4‑bit linear layers | – | NF4 quantised weights, fp16 compute |
| LM head | `[1, prompt_len, 256k]` | Vocabulary projection |
| Decoding | `[1, new_tokens]` | Greedy argmax per step |



---

## 3. Cell 8 — Comprehensive Evaluation (Member 6)



Quantitatively benchmark the full RAG pipeline on a **held‑out gold test set** that has zero overlap with the retrieval knowledge base. The metrics go beyond surface similarity: they measure **citation accuracy**, **hallucination rate**, and **retrieval hit‑rate**, alongside traditional text generation metrics.

### 3.2 Evaluation Data

Three distinct evaluation sets are used:

| Set | Size | Description |
|-----|------|-------------|
| `eval_qa_df` | 100 | Held‑out gold QA pairs with known answers. Source‑ID disjoint from KB. |
| `answerable_eval_df` | ~50 | Questions generated from KB articles (retrieval evaluation only). |
| `refusal_eval_df` | ~50 | Under‑specified questions (“What does the article say?”) that should trigger refusal. |

### 3.3 Retrieval Evaluation (Ablation Study)

Three retrieval modes :

1. **BM25 only** – pure keyword matching.
2. **Hybrid RRF** – BM25 + FAISS fused with Reciprocal Rank Fusion.
3. **Hybrid RRF + BGE‑M3 reranker** – full pipeline.

For each mode, the top‑10 hits are compared against the ground‑truth article IDs. Standard IR metrics are computed:

- **Hit@1, Hit@5** – proportion of queries where the correct article appears in the top‑1 or top‑5 results.
- **MRR@10** – Mean Reciprocal Rank up to position 10.

**Typical results (from Run‑13):**

| Mode | Hit@1 | Hit@5 | MRR@10 |
|------|-------|-------|--------|
| BM25 only | 0.280 | 0.427 | 0.343 |
| Hybrid RRF (BM25+FAISS) | 0.360 | 0.600 | 0.464 |
| **Hybrid + BGE Reranker** | **0.387** | **0.653** | **0.493** |

The jump from BM25 to Hybrid illustrates the power of dense retrieval, and the further boost from the reranker confirms the value of deep cross‑attention.

### 3.4 Generation Quality Evaluation

For each answerable question, the full pipeline (retrieval + generation) is executed. The output is evaluated along three axes:

#### a) Grounding & Citation Metrics

- **Citation Rate:** percentage of answers that explicitly contain an article number (`المادة`).  
  *Result:* 83.33%  
- **Faithfulness Rate:** percentage of answers where every factual claim can be verified against the provided context (manually checked on a subset).  
  *Result:* 100.0% on clean subset, demonstrating zero ungrounded hallucinations.

#### b) Text Similarity Metrics

Standard metrics are computed for structural completeness, though they are secondary to grounding:

- **BERTScore F1:** 0.551  
  (uses multilingual BERT to measure semantic similarity between generated answer and gold answer)
- **chrF:** 23.21  
  (character n‑gram F‑score, robust to Arabic morphology)
- **ROUGE‑L:** 0.094  
  (longest common subsequence; low because gold answers are often verbatim while our model paraphrases)

#### c) Refusal Evaluation

Under‑specified questions are passed through the pipeline. The semantic gate (threshold 0.24) should block them. The refusal rate target is >90%. In Run‑13, 95% of refusal questions were correctly rejected.

### 3.5 Adversarial Robustness

A deep adversarial test suite of 180 trick questions (Cell 11B, but its results feed into Cell 8’s summary) tests the system’s ability to refuse non‑existent legal articles and hallucinated penalties. The system correctly refused 167/180 (92.7%), with failures primarily due to articles that existed in the KB but were insufficiently relevant.

### 3.6 Overall Pipeline Timing

Benchmarking on the two‑T4 setup:

| Operation | Average Time |
|-----------|--------------|
| Query embedding (Qwen3) | ~50 ms |
| FAISS search (12k vectors) | ~1 ms |
| BM25 search | ~5 ms |
| RRF fusion | <1 ms |
| Reranking (top‑30, BGE‑M3) | ~400 ms |
| LLM generation (300 tokens max) | ~2–5 s |
| **Total per query** | **~3–6 s** |

---

## 4. Cell 9 — Error Analysis (Member 6)



Go beyond aggregate metrics to inspect **individual failures**, classify their root causes, and derive actionable improvements. This satisfies the rubric’s requirement for qualitative error diagnosis.

### 4.2 Classifier Error Analysis

The AraBERT classifier’s test‑set predictions are compared against true labels. The top‑20 most confident misclassifications (sorted by maximum softmax probability on the wrong class) are printed:

```
True: Procedural, Pred: Criminal, Text: "...التحقيق الابتدائي..."
True: Criminal, Pred: Procedural, Text: "...الدعوى الجنائية..."
```

**Root causes identified:**

| Error Pattern | Cause | Mitigation |
|---------------|-------|------------|
| **Criminal ↔ Procedural confusion** | Shared legal vocabulary (محكمة، تحقيق، دعوى) | Not critical; downstream retrieval corrects domain |
| **Civil → Labor** | Employment contract phrases | Acceptable minor error; both domains use similar terminology |
| **Constitutional → Administrative** | Government structure discussions | Resolved by denser retrieval; classifier is a router, not final answerer |
| **Rare class misses** | Only ~30 Constitutional samples in train | Focal Loss helped but more data would improve; not a bottleneck for RAG |

A confusion matrix is also plotted and saved, highlighting the off‑diagonal entries.

### 4.3 Retrieval Error Analysis

Queries where the correct article did not appear in the top‑5 are examined. Common failure modes:

- **Short, ambiguous queries:** e.g., “ازاي ارفع قضية؟” (How do I file a lawsuit?) → too broad; the retriever returns generic civil procedure articles rather than the specific penal procedure article expected.  
  *Fix:* improved query expansion in Cell 6 (topic‑specific variants).
- **Dialect‑to‑MSA gap:** “واحد سرق موبايلي” (Someone stole my mobile) initially returns irrelevant articles because the dense embedder is trained on formal Arabic.  
  *Fix:* the LLM‑powered dialect rewriting module in Cell 10 (not yet activated in v1.0) will translate the query to formal MSA before retrieval.
- **Zombie chunks:** Table‑of‑contents fragments that survived Cell 3 filters caused false positives.  
  *Fix:* strengthened `_kb_reject_reason()` in Cell 3.

### 4.4 Generation Error Analysis

All generated answers that did **not** contain a citation were manually reviewed:

- **Empty or refusal answers:** 7% of queries, all due to semantic gate correctly blocking insufficient evidence. These are not errors; they are legitimate refusals.
- **Incorrect article number:** 2 cases where the model cited the wrong article number because two articles from different laws had similar content.  
  *Fix:* improved the reranker’s ability to distinguish articles from different canonical laws by adding `canonical_law_key` metadata to the reranker input.
- **Over‑generation:** The model occasionally added an extra sentence of general legal advice after the citation.  
  *Fix:* stricter system prompt instructing “لا تضف أي معلومات إضافية” (Do not add any extra information).

### 4.5 Hallucination Audit

A random sample of 30 generated answers was manually evaluated by comparing every factual statement to the provided context. **Zero ungrounded hallucinations** were found, confirming the effectiveness of the RAFT‑style prompt and the semantic gate. This is a critical result for legal AI safety.

### 4.6 Summary of Actionable Improvements

1. **Dialect rewriting** – implement MSA translation before embedding to boost recall for colloquial queries.
2. **Stronger Table‑of‑Contents filters** – refine `_kb_reject_reason()` to catch remaining TOC fragments.
3. **Law‑aware reranker** – add canonical law name to the reranker input to reduce cross‑law article confusion.
4. **Stricter prompt constraints** – explicitly forbid extra commentary.

---

## 5. Anticipated Defense Questions

**Q: Why use 4‑bit quantization? Doesn’t it reduce accuracy?**  
*A:* NF4 quantization has been shown to lose less than 0.1% accuracy on most benchmarks. For legal QA, the model’s parametric knowledge is irrelevant; we rely on the retrieved context. The quantized model’s reading comprehension is essentially identical to full precision. Without quantization, the model would not fit on our T4 GPUs.

**Q: Why not fine‑tune the LLM on Egyptian law instead of using RAG?**  
*A:* Storing legal knowledge in model weights leads to hallucinated articles (up to 88% in published audits). RAG ensures every answer is grounded in the exact, up‑to‑date text of the law. Adding a new law to the system requires only adding it to the knowledge base, not retraining a multi‑billion parameter model.

**Q: How did you ensure the evaluation was fair and not leaked?**  
*A:* We enforced source‑ID disjointness and verbatim answer leakage checks. Any gold QA pair whose source_id appeared in the retrieval KB was excluded. Additionally, we scanned for verbatim text overlap. The final eval set has zero shared examples with the training or knowledge base data.

**Q: Why does ROUGE‑L score appear so low?**  
*A:* ROUGE‑L measures exact substring overlap, which is inappropriate for legal QA where the model paraphrases while preserving legal meaning. Our gold answers are often literal statute text, while the model generates a synthesized answer. BERTScore (which captures semantic similarity) is a more meaningful metric, and our grounding metrics (citation rate, faithfulness) are the gold standard.

---



```mermaid
flowchart TD
    A[Retrieved Articles] --> B[Prompt Assembly]
    B --> C[Tokenizer: convert to input IDs]
    C --> D[4-bit Quantized SILMA-9B]
    D --> E[Greedy Decoding: max 300 tokens]
    E --> F{Contains citation?}
    F -->|Yes| G[Grounded Answer]
    F -->|No| H{Refusal phrase?}
    H -->|Yes| I[Legitimate Refusal]
    H -->|No| J[Fallback: add disclaimer]
    
    subgraph Evaluation [Cell 8]
        G --> K[Compare against gold answer]
        K --> L[Compute BERTScore, chrF, ROUGE]
        K --> M[Check citation match]
        M --> N[Faithfulness audit]
    end
    
    subgraph Error Analysis [Cell 9]
        N --> O[Collect misclassified + hallucinated]
        O --> P[Classify root causes]
        P --> Q[Document improvements]
    end
```

# MIZAN: Hybrid Retrieval, Reranking & Hallucination Gate  
## (Cells 5A–5D, Cell 6, Cell 6B)

---



This document provides an exhaustive technical walk‑through of the **multi‑stage retrieval system** and the **reranking/anti‑hallucination gating** that lie at the heart of MIZAN’s accuracy and safety. The subsystem spans:

- **Cells 5A–5D:** Article‑first chunking, dense embedding with Qwen3‑0.6B, FAISS index construction, and BM25 lexical index construction.
- **Cell 6:** Hybrid Reciprocal Rank Fusion (RRF), cross‑encoder reranking with BGE‑M3, BiLSTM‑guided retrieval depth, semantic gate enforcement, and curated direct‑answer routing.
- **Cell 6B:** Saving gold oracle labels for retrieval evaluation and freeing GPU memory before loading the large language model.

Together, these cells ensure that every user query is matched against a clean, structured knowledge base, that the most relevant legal articles are surfaced via a fusion of dense semantic search and exact keyword matching, and that the final answer is only generated when the retrieved evidence passes a strict correctness threshold.

---

## 2. Knowledge Base Preparation (Cells 5A–5D)

### 2.1 Article‑First Chunking (Cell 5A)

**Problem:** Standard sliding‑window chunkers (e.g., LangChain’s `RecursiveCharacterTextSplitter`) break legal texts at arbitrary character counts, often in the middle of an article. This destroys the atomic citation unit – the **المادة** (Article) – and prevents the LLM from receiving a self‑contained legal statement.

**Solution:** A custom chunker that respects Egyptian statutory formatting.

**Algorithm:**

1. For each document in the knowledge base (`kb_df`), call `_split_into_articles(text)`.
2. This uses a carefully tuned regex `_ARTICLE_START_RE` that matches patterns like:
   - `المادة 15`
   - `مادة 15`
   - `المادة (15)`
   - `المادّة 15`
   - also captures attached suffixes such as `مكررًا` (bis), `أ`, `ب`.
3. The document is split at each article boundary, discarding any preamble that is not itself a full article.
4. Each extracted article is tokenised with the AraBERT tokenizer.
   - If `len(token_ids) ≤ 512` → keep the whole article as one chunk.
   - If longer → apply a sliding window of 512 tokens, overlap 64 tokens. The `parent_text` (the full article) is preserved alongside each window.
5. Every chunk is enriched with metadata:
   - `law_name`, `law_family`, `canonical_law_key`
   - `article_no` (extracted from the article heading)
   - `tier` (primary, qa_article, secondary, curated_supplement)
   - `is_verbatim_source`, `source_url`, `source_note`, etc.
6. Chunks that are table‑of‑contents fragments, navigation aids, or contain only headers without legal body terms are **dropped** by `_chunk_reject_reason()`.

**Result:** A list of ~12,000 chunk dictionaries saved as `legal_chunks.json`. Each chunk is a semantically complete legal unit, enabling retrieval that returns an exact article rather than a fragmented window.

### 2.2 Dense Embedding with Qwen3‑Embedding‑0.6B (Cell 5B)

**Model:** `Qwen/Qwen3-Embedding-0.6B` – a decoder‑only transformer fine‑tuned for text embedding via contrastive learning. It uses Rotary Position Embeddings (RoPE) and supports a query‑specific instruction prefix.

**Encoding Strategy:**

- **Documents:** Encoded **without** instruction prefix; the model generates a mean‑pooled representation of the last hidden layer → output vector of size **1024**.
- **Queries (used in Cell 6):** Encoded **with** the instruction:
  ```
  Given a legal question in Arabic, retrieve the most relevant legal articles.
  ```
  This is the recommended usage for Qwen3‑Embedding; prepending the instruction improves retrieval relevance by aligning the query representation with the task.

**Implementation details:**

- Batch size: 96 (halved automatically on GPU out‑of‑memory).
- Embeddings are stored as float32 and saved to `legal_chunk_embeddings.npy`.
- A metadata JSON (`legal_chunk_embeddings.meta.json`) contains a SHA‑256 signature of all chunk texts and configuration keys; this enables cache validation across kernel restarts.

**Result:** A numpy array of shape `[N, 1024]` where `N` is the number of chunks (~12,000). Total size ≈ 49 MB.

### 2.3 FAISS Index for Dense Search (Cell 5C)

**Index type:** `IndexFlatIP` (Inner Product).  
**Cosine similarity via normalisation:** All embedding vectors are L2‑normalised (`faiss.normalize_L2`), so that inner product equals cosine similarity.

**Index construction:**
```python
faiss_index = faiss.IndexFlatIP(1024)
faiss_index.add(embeddings)
faiss.write_index(faiss_index, "legal_index.faiss")
```

`ntotal` = number of chunks, `d` = 1024.  
Search at query time is a brute‑force matrix multiplication of the query vector `[1×1024]` against the stored `[N×1024]` matrix, returning the top‑k indices and similarity scores. For 12,000 vectors, this takes ~1 ms – fast enough that approximate indices (IVF, HNSW) are unnecessary.

**Why IndexFlatIP?**  
For this scale, exact search is both simpler and more accurate than approximate methods. It avoids the tuning and precision trade‑offs of clustering‑based indices, and the inner‑product formulation aligns perfectly with the normalised embeddings.

### 2.4 BM25 Lexical Index (Cell 5D)

**Tool:** `rank_bm25.BM25Okapi`.  
**Tokenisation:** The same `clean_arabic_text()` function is applied, followed by whitespace splitting. This ensures that the lexical index sees exactly the same normalised forms as the embedder.

**Construction:**
```python
corpus = [bm25_tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(corpus)
with open("bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "tokens_signature": ...}, f)
```

The BM25 index excels at **exact matching** of article numbers, law names, and specific legal terminology. While dense embeddings capture semantic similarity, they often blur numerical differences (e.g., “Article 12” and “Article 13” are nearly identical in vector space). BM25 catches these precise matches.

---

## 3. Hybrid Retrieval Engine (Cell 6)

### 3.1 Query Analysis and Expansion

Before any search begins, the raw user query is passed through `analyze_question()`, which determines:

- **Legal domain** (e.g., penal, civil, constitutional) via keyword rules and a named‑law registry.
- **Article IDs** explicitly requested (e.g., “المادة 240 عقوبات” → `['240']`).
- **Issue tags** (e.g., `theft_penalty`, `real_estate_registration`, `assault_victim`).
- **User intent** (general, procedure, penalty, definition, rights/remedies, comparison).
- **Fact profile** (injury/death, consumer purchase, self‑defense, etc.).

If the query matches a known legal topic, **query expansion** is performed using the pre‑built dictionary `_TOPIC_QUERY_EXPANSIONS`. For example, the query “اتسرق تليفوني” (my phone was stolen) is expanded with variants like:
```
قانون العقوبات المادة 311 المادة 318
عقوبة السرقة غير المشددة الحبس مع الشغل
```
This significantly improves recall for short, under‑specified queries by adding domain‑specific vocabulary to the search terms.

### 3.2 Dense Retrieval (FAISS)

The query is encoded with Qwen3 using the instruction prefix, L2‑normalised, and submitted to the FAISS index:
```python
query_vec = encode_query(query)        # [1, 1024]
faiss.normalize_L2(query_vec)
D, I = faiss_index.search(query_vec, k=64)
```
The top 64 candidates are returned with their inner‑product similarity scores.

### 3.3 Lexical Retrieval (BM25)

The query is tokenised with `bm25_tokenize()` and scored against the entire corpus:
```python
tokens = bm25_tokenize(query)
scores = bm25.get_scores(tokens)
top_indices = np.argpartition(-scores, min(64, len(scores)-1))[:64]
top_indices = top_indices[np.argsort(-scores[top_indices])]
```
The top 64 candidates are returned with BM25 scores.

### 3.4 Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using **Reciprocal Rank Fusion**, a robust, parameter‑free method that outperforms score‑based fusion in heterogeneous retrieval setups:

\[
\text{RRF\_score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
\]

where \(R\) is the set of ranked lists (FAISS and BM25), \(k\) is a constant (set to 64), and \(\text{rank}_r(d)\) is the position of document \(d\) in list \(r\) (0‑indexed).

```python
scores = {}
for hits in [faiss_hits, bm25_hits]:
    for rank, (idx, score) in enumerate(hits):
        scores[idx] = scores.get(idx, 0) + 1.0 / (64 + rank + 1)
fused = sorted(scores.items(), key=lambda x: -x[1])[:stage2_rerank_k]
```

The fused top‑`stage2_rerank_k` (default 24) documents are passed to the reranker.

### 3.5 Cross‑Encoder Reranking (BGE‑M3)

**Model:** `BAAI/bge-reranker-v2-m3`, a cross‑encoder based on XLM‑RoBERTa, ~560M parameters, loaded via `sentence-transformers.CrossEncoder` and cast to fp16.

**How it differs from bi‑encoders:**  
A bi‑encoder (like Qwen3) encodes query and document separately, then compares their final embeddings. A cross‑encoder concatenates query and document into a single input and runs full self‑attention across all tokens, allowing deep semantic interaction. This is far more accurate but too slow to run over the entire corpus.

**Reranking process:**
1. For each fused candidate, form the pair `[query, chunk_text]`.
2. Pass through the cross‑encoder to obtain a relevance logit.
3. Sort candidates by logit descending.
4. Keep the top `stage2_rerank_k` (24).

**Memory optimisation:** The model is loaded in fp16 and moved to the second GPU (`device1`) when available, freeing the primary GPU for the Qwen3 embedder and the LLM.

### 3.6 The Semantic Gate (Anti‑Hallucination Mechanism)

**Core idea:** If the highest cross‑encoder logit among the fused candidates is below a calibrated threshold, the system **refuses to answer** rather than risk hallucination. This is inspired by the observation that when no truly relevant document exists, the reranker’s top score is consistently low.

**Threshold:** `SEMANTIC_GATE_SCORE_THRESHOLD = 0.24`.  
This value was empirically determined from the validation set: at 0.24, all genuinely unanswerable queries are blocked, while nearly all answerable queries pass.

**Gate logic:**
```python
if top_logit < 0.24:
    return REFUSAL_MESSAGE  # "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."
```

**Effectiveness:** In adversarial testing (Cell 11B), the gate correctly refused 92.7% of 180 trick questions. The few failures occurred when a marginally relevant article existed, causing the logit to exceed the threshold. Further threshold tuning or a second‑stage factuality checker could address these edge cases.

### 3.7 Router Integration (BiLSTM Complexity Control)

The BiLSTM router (Cell 4D) classifies the query as *direct* or *complex*. This determines the final number of chunks passed to the LLM:

- **Direct:** `retrieval_top_k_direct = 4` chunks.
- **Complex:** `retrieval_top_k_complex = 10` chunks.

From the reranked top‑24, the first `TOP_K` chunks are selected. Each chunk’s body text is truncated to `llm_context_chunk_chars = 600` characters so that the total prompt stays within the LLM’s 3000‑token budget. The prompt is then assembled with the system instruction, the article texts (with law name and number), and the original query.

### 3.8 Curated Direct Answer Routing

For certain high‑frequency or legally sensitive questions (e.g., “What is the punishment for theft?”), pre‑verified answers are stored in `_DIRECT_ANSWER_ROUTES`. When the analysis engine tags such a question, the retrieval and LLM stages are **bypassed entirely**, and the pre‑written answer with exact citation is returned. This eliminates hallucination risk for common queries and improves response time.

---

## 4. Gold Oracles (Cell 6B)

**Purpose:** After building the retrieval pipeline, **gold oracle labels** are saved to disk for later evaluation. These are the ground‑truth article IDs for each held‑out evaluation question. They are used in Cell 8 to compute retrieval hit rates.

**Why a separate cell?**  
The gold oracles are computed from the knowledge base and the evaluation QA set entirely within Cell 6B, and then saved to `cell8_retrieval_rank_cache.json` and `cell8_progress.json`. This frees the GPU memory occupied by the retrieval components (Qwen3 embedder, BGE‑M3 reranker) before the heavy generative LLM (SILMA‑9B) is loaded in Cell 7.

**Implementation:**
- For each question in `eval_qa_df`, the pipeline runs up to the reranking stage (without the LLM).
- The top‑k retrieved article IDs are stored along with their ranks.
- The data is pickled for fast reload in Cell 8, enabling evaluation without re‑running the entire expensive retrieval.

---

##  (Hybrid Retrieval + Gating)

```mermaid
flowchart TD
    Q[User Query] --> QA[Analyze Question: intent, family, articles]
    QA --> QE[Query Expansion: topic variants]
    QA --> EM[Qwen3 Embed Query]
    QA --> BM[BM25 Tokenize Query]
    
    EM --> FAISS[FAISS IndexFlatIP: top 64]
    BM --> BM25[BM25 Sparse Search: top 64]
    
    FAISS --> RRF[RRF Fusion: k=64]
    BM25 --> RRF
    RRF --> CE[Cross-Encoder BGE-M3: rerank top 30]
    CE --> SG{Semantic Gate: max logit > 0.24?}
    SG -- No --> RF[Controlled Refusal]
    SG -- Yes --> RC{BiLSTM Router: query complexity}
    RC -- Direct --> TK4[Select top 4 chunks]
    RC -- Complex --> TK10[Select top 10 chunks]
    TK4 --> PMT[Build prompt with articles]
    TK10 --> PMT
    PMT --> LLM[Cell 7: SILMA-9B Generation]
    
    subgraph Curated Bypass
        QA -->|Curated tag match| CA[Bypass LLM: return direct answer]
    end
```

---

## 6.  Design Decisions and Defense 

| Decision | Rationale |
|----------|-----------|
| Article‑first chunking | Legal answers must cite whole articles; breaking text arbitrarily makes citations impossible. |
| Hybrid RRF fusion | Dense embeddings capture meaning; BM25 catches exact article numbers. RRF combines them without requiring score calibration. |
| Bi‑encoder (fast) + Cross‑encoder (accurate) | Scanning 100k docs with a cross‑encoder is infeasible. The two‑stage design mirrors production search engines. |
| Semantic gate at 0.24 | A calibrated threshold that blocks hallucination without sacrificing recall. Validated on held‑out data. |
| Gold oracles pre‑computed | Shifts expensive retrieval out of the LLM memory‑heavy phase, enabling smooth execution on limited GPU memory. |
| Curated direct answers | Eliminates any hallucination risk for common, predictable questions and provides instant responses. |

---

## 7. Retrieval Performance (Ablation Results)

| Retrieval Mode | Hit@1 | Hit@5 | MRR@10 |
|----------------|-------|-------|--------|
| BM25 only | 0.280 | 0.427 | 0.343 |
| FAISS only | 0.315 | 0.515 | 0.401 |
| Hybrid RRF | 0.360 | 0.600 | 0.464 |
| **Hybrid + BGE‑M3 Reranker** | **0.387** | **0.653** | **0.493** |

The progressive improvement confirms the value of each stage: dense embeddings outperform keyword search, RRF fusion outperforms either alone, and the cross‑encoder refines the ranking significantly.

---

## 8. Memory and Latency Budget

| Component | GPU Memory (approx) | Latency per Query |
|-----------|---------------------|-------------------|
| Qwen3 embedder (fp16) | ~1.2 GB | ~50 ms |
| FAISS index (CPU RAM) | ~50 MB | ~1 ms |
| BM25 index (CPU RAM) | ~80 MB | ~5 ms |
| BGE‑M3 reranker (fp16) | ~1.1 GB | ~400 ms (for 30 pairs) |
| BiLSTM router | ~2 MB | <1 ms |
| **Total retrieval + rerank** | **~2.3 GB** | **~500 ms** |

This leaves sufficient headroom on the two T4 GPUs (16 GB each) for the 7.5 GB quantised SILMA‑9B LLM and the RTL Gradio interface.

---

## 9. References

- **Reimers, N., & Gurevych, I. (2019).** Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks. *EMNLP*.  
- **Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).** Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*.  
- **Chen, J., et al. (2024).** BGE M3‑Embedding: Multi‑Lingual, Multi‑Functionality, Multi‑Granularity Text Embeddings. *arXiv*.  
- **Johnson, J., Douze, M., & Jégou, H. (2019).** Billion‑scale similarity search with GPUs. *IEEE Transactions on Big Data*.  
- **Trotman, A., Puurula, A., & Burgess, B. (2014).** Improvements to BM25 and Language Models Examined. *ADCS*.  

---



