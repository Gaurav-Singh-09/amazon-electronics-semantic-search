# Amazon Electronics Semantic Search Engine (FAISS + CUDA)

A dense vector semantic search and budget-filtered recommendation system indexing **526,489 Amazon electronics products** using `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS` inner-product (cosine similarity) search.

---

## Project Overview
- **Raw Dataset:** 1,610,012 Amazon Electronics metadata records (Parquet shards)
- **Cleaned Catalog:** 526,489 valid products with price normalization, direct Amazon product URLs, and composite search text (`Title` + `Specs` + `Category`)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized float32 vectors) accelerated on **NVIDIA CUDA**
- **Vector Index:** `faiss.IndexFlatIP` (exact cosine similarity over 526,489 normalized vectors)
- **Retrieval Pipeline:** Top-$k$ semantic candidate retrieval (`fetch_k = k * 10`) paired with a strict maximum price ceiling (`max_budget`) filter

---

## Sample Query Output
**Query:** `"wireless noise cancelling over ear headphones"` | **Budget Cap:** `$50.00`

1. **Wireless Headphones Over Ear for Active Noise Cancelling Headphones** — `$49.99` | Rating: `4.1` | Similarity: `0.7757`
2. **Qisebin E7 Active Noise Cancelling Headphones...** — `$40.22` | Rating: `4.5` | Similarity: `0.7219`
3. **Vsonus Noise Cancelling Headphones Wireless Bluetooth...** — `$29.99` | Rating: `3.6` | Similarity: `0.7062`

---

## Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, NumPy, PyArrow / Parquet
- **Embeddings & GPU Acceleration:** PyTorch (CUDA), Hugging Face `sentence-transformers`
- **Vector Search:** Meta `FAISS` (`faiss-cpu` / `faiss-gpu`)

---

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/Gaurav-Singh-09/amazon-electronics-semantic-search.git](https://github.com/Gaurav-Singh-09/amazon-electronics-semantic-search.git)
   cd amazon-electronics-semantic-search
