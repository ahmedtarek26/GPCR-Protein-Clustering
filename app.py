"""
GPCR Protein Classification — Streamlit Web App
================================================
Enter a protein sequence and ask any biological question.
The app embeds the sequence with ESM-2, retrieves the most similar
known GPCRs, and feeds the cluster knowledge base into TinyLlama to
answer your question.
"""

import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, EsmModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"

MULTIMODAL_CSV = DATA_DIR / "final_multimodal_clusters.csv"
SEQ_EMB_CSV    = DATA_DIR / "embedding_sequences_mean_pooling.csv"

ESM_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data & building knowledge base…")
def load_data():
    multimodal_df = pd.read_csv(MULTIMODAL_CSV)
    known_ids = multimodal_df["ID"].tolist()

    seq_df = pd.read_csv(SEQ_EMB_CSV)
    if "uniref_id" in seq_df.columns:
        seq_df = seq_df.rename(columns={"uniref_id": "ID"})

    pure_seq_df = seq_df[seq_df["ID"].isin(known_ids)].copy()
    pure_seq_df = pure_seq_df.set_index("ID").reindex(known_ids)
    emb_cols = [c for c in pure_seq_df.columns if c != "ID"]
    known_vecs = pure_seq_df[emb_cols].values

    # Build cluster knowledge base
    cluster_kb = {}
    for cluster_id in sorted(multimodal_df["super_cluster"].unique()):
        cluster_proteins = multimodal_df[multimodal_df["super_cluster"] == cluster_id]
        names, keywords, functions, subfamilies, go_terms = [], [], [], [], []

        for _, row in cluster_proteins.iterrows():
            if pd.notna(row.get("Protein names", "")):
                n = str(row["Protein names"]).strip()
                if n:
                    names.append(n[:120])
            if pd.notna(row.get("Keywords", "")):
                keywords.extend([k.strip() for k in str(row["Keywords"]).split(";") if k.strip()])
            if pd.notna(row.get("Function [CC]", "")):
                f = str(row["Function [CC]"]).strip()
                if f:
                    functions.append(f[:300])
            if pd.notna(row.get("subfamily", "")):
                s = str(row["subfamily"]).strip()
                if s:
                    subfamilies.append(s)
            if pd.notna(row.get("Gene Ontology (GO)", "")):
                go_terms.extend([g.strip() for g in str(row["Gene Ontology (GO)"]).split(";") if g.strip()])

        cluster_kb[int(cluster_id)] = {
            "size": len(cluster_proteins),
            "subfamilies": list(dict.fromkeys(subfamilies))[:6],
            "keywords": list(dict.fromkeys(keywords))[:20],
            "sample_functions": list(dict.fromkeys(functions))[:3],
            "sample_names": list(dict.fromkeys(names))[:5],
            "go_terms": list(dict.fromkeys(go_terms))[:10],
        }

    return multimodal_df, known_ids, known_vecs, cluster_kb


@st.cache_resource(show_spinner="Loading ESM-2 protein encoder…")
def load_esm():
    tok = AutoTokenizer.from_pretrained(ESM_MODEL_ID)
    mdl = EsmModel.from_pretrained(
        ESM_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    if DEVICE.type != "cuda":
        mdl = mdl.to(DEVICE)
    mdl.eval()
    return mdl, tok


@st.cache_resource(show_spinner="Loading TinyLlama language model…")
def load_llm():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    if DEVICE.type != "cuda":
        mdl = mdl.to(DEVICE)
    mdl.eval()
    return mdl, tok


# ── Core pipeline functions ────────────────────────────────────────────────────

def clean_sequence(raw: str) -> str:
    seq = raw.replace("-", "").replace(".", "").replace(" ", "").replace("\n", "")
    return seq.upper()


def embed_sequence(raw_sequence: str, esm_model, esm_tokenizer) -> np.ndarray:
    seq = clean_sequence(raw_sequence)[:1022]
    inputs = esm_tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    token_embs = outputs.last_hidden_state[0, 1:-1, :]
    vec = token_embs.mean(dim=0).float().cpu().numpy()
    return vec.reshape(1, -1)


def retrieve_neighbors(query_vec, known_vecs, known_ids, multimodal_df, k=3):
    sims = cosine_similarity(query_vec, known_vecs)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [
        {
            "rank": i + 1,
            "idx": int(top_k[i]),
            "id": known_ids[top_k[i]],
            "similarity": float(sims[top_k[i]]),
            "data": multimodal_df.iloc[top_k[i]],
        }
        for i in range(k)
    ]


def build_rag_context(neighbors, cluster_kb):
    clusters = [int(n["data"]["super_cluster"]) for n in neighbors]
    vote_counts = Counter(clusters)
    voted_cluster = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[voted_cluster] / len(clusters)

    fallback_used = False
    if voted_cluster == -1:
        non_noise = [c for c in clusters if c != -1]
        if non_noise:
            voted_cluster = Counter(non_noise).most_common(1)[0][0]
            confidence = non_noise.count(voted_cluster) / len(clusters)
            fallback_used = True

    kb = cluster_kb.get(voted_cluster, {})
    label = "NOISE / unclassified" if voted_cluster == -1 else f"Cluster {voted_cluster}"

    lines = [f"=== PREDICTED GPCR CLUSTER: {label} ==="]
    if fallback_used:
        lines.append(
            f"(Note: majority vote was cluster -1/noise; using best non-noise cluster "
            f"from neighbors — {confidence:.0%} support)\n"
        )
    else:
        lines.append(
            f"(Majority vote: {vote_counts[voted_cluster]}/{len(clusters)} neighbors "
            f"agree — {confidence:.0%} confidence)\n"
        )

    lines += [
        "--- CLUSTER BIOLOGICAL PROFILE ---",
        f"Keywords     : {', '.join(kb.get('keywords', [])[:12]) or 'N/A'}",
        f"GO terms     : {', '.join(kb.get('go_terms', [])[:8]) or 'N/A'}",
        f"Cluster size : {kb.get('size', '?')} proteins",
        "",
        "--- KNOWN FUNCTIONS IN THIS CLUSTER ---",
    ]
    for fn in kb.get("sample_functions", [])[:2]:
        lines.append(f"  • {fn[:400]}")

    lines += ["", "--- TOP 3 MOST SIMILAR KNOWN PROTEINS ---"]
    for n in neighbors:
        d = n["data"]
        name   = str(d.get("Protein names", "Unknown"))[:80]
        func   = str(d.get("Function [CC]", "N/A"))[:250]
        kw     = str(d.get("Keywords", "N/A"))[:120]
        raw_cluster = int(d.get("super_cluster", -1))
        cluster_note = "(noise label)" if raw_cluster == -1 else f"cluster {raw_cluster}"
        lines += [
            f"\n  [{n['rank']}] {n['id']}  (cosine similarity: {n['similarity']:.4f})  [{cluster_note}]",
            f"      Name     : {name}",
            f"      Keywords : {kw}",
            f"      Function : {func}",
        ]

    return "\n".join(lines), voted_cluster, confidence


def llm_answer(context: str, question: str, neighbors: list, cluster: int,
               confidence: float, cluster_kb: dict,
               llm_model, llm_tokenizer, max_new_tokens: int = 250) -> str:
    """
    Ask TinyLlama for a concise interpretation paragraph.
    We pass a compact, focused prompt — not the full raw context dump.
    """
    kb = cluster_kb.get(cluster, {})
    cluster_label = "NOISE / unclassified" if cluster == -1 else f"Cluster {cluster}"
    kws       = ', '.join(kb.get('keywords', [])[:10]) or 'N/A'
    top_names = [str(n['data'].get('Protein names', ''))[:80] for n in neighbors[:3]]
    top_sims  = [n['similarity'] for n in neighbors[:3]]
    functions = kb.get('sample_functions', [])
    fn_snip   = functions[0][:250] if functions else 'No function annotation available.'

    compact_context = (
        f"Predicted GPCR group: {cluster_label} (confidence: {confidence:.0%})\n"
        f"Key biological keywords: {kws}\n"
        f"Known function example: {fn_snip}\n"
        f"Top-3 nearest known proteins (by sequence similarity):\n"
        + "\n".join(
            f"  {i+1}. {top_names[i]} (similarity {top_sims[i]:.3f})"
            for i in range(len(top_names))
        )
    )

    system_msg = (
        "You are an expert AI assistant specialising in GPCR (G protein-coupled receptor) biology.\n"
        "Given a brief classification summary of a query protein, write a clear, informative "
        "paragraph that:\n"
        "1. Describes its likely biological role and signalling mechanism.\n"
        "2. Mentions the most similar known proteins.\n"
        "3. Notes any relevant pharmacological or disease significance if inferable.\n"
        "Write in a professional but accessible style, like a knowledgeable AI assistant."
    )

    user_msg = (
        f"Classification summary:\n{compact_context}\n\n"
        f"User question: {question}\n\n"
        "Please provide a thorough, well-structured answer."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(next(llm_model.parameters()).device)
    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return llm_tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GPCR Protein Classifier",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GPCR Protein Classification")
st.markdown(
    "Enter an amino-acid sequence and a question. "
    "The app embeds your sequence with **ESM-2**, retrieves the nearest known GPCRs, "
    "and answers your question using **TinyLlama** grounded in the cluster knowledge base."
)

# ── Load everything (cached after first run) ───────────────────────────────────
multimodal_df, known_ids, known_vecs, cluster_kb = load_data()
esm_model, esm_tokenizer = load_esm()
llm_model, llm_tokenizer = load_llm()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    k_neighbors = st.slider("Number of nearest neighbors (k)", min_value=1, max_value=10, value=3)
    max_tokens   = st.slider("Max new tokens for LLM", min_value=100, max_value=600, value=350, step=50)
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE.type.upper()}`")
    if DEVICE.type == "cuda":
        st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
    st.markdown(f"**Known proteins:** {len(known_ids)}")
    st.markdown(f"**Clusters:** {len(cluster_kb)}")
    st.markdown("---")
    st.markdown("**ESM-2 model:** `facebook/esm2_t33_650M_UR50D`")
    st.markdown("**LLM:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`")

# ── Main form ──────────────────────────────────────────────────────────────────
DEMO_SEQ = (
    "QILWSIIFGTMVFVAAGGNIIVIWIVLTNKRMRTVTNYFLVNLSVADIMVSTLNVIFNFIYMLNSNWPFGE"
    "LFCKITNFIAILSVGASVFTLMAISIDRYLAIVHPLRPRMSRTATVIIIVIIWIASSFLSLPNIICSKTIVE"
    "EFKNGDSRVVCYLEWYDGISTKSRIEYIYNVIILLVTYMLPIASMSYTYFRVGRELWGSQSIGECTAKQMES"
    "IKSKRKIVKMMMIVVAIFGVCWAPYHIYFLLAHHYPQIINSKYVQHTYLTIYWLAMSNSVYNPFVYCWMNSR"
    "FRQGFRNIFCC"
)

col1, col2 = st.columns([3, 2])

with col1:
    sequence_input = st.text_area(
        "Amino-acid sequence",
        value=DEMO_SEQ,
        height=200,
        placeholder="Paste your protein sequence here…",
    )

with col2:
    question_input = st.text_area(
        "Your question",
        value="What GPCR family and subfamily does this protein belong to, and what is its likely function?",
        height=200,
        placeholder="Ask biological question about this sequence…",
    )

run_button = st.button("🔍 Analyse Sequence", type="primary", use_container_width=True)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_button:
    raw_seq = sequence_input.strip()
    question = question_input.strip()

    if not raw_seq:
        st.error("Please enter a protein sequence.")
        st.stop()
    if not question:
        st.error("Please enter a question.")
        st.stop()

    cleaned = clean_sequence(raw_seq)
    if len(cleaned) < 10:
        st.error(f"Sequence too short after cleaning ({len(cleaned)} AAs). Please check your input.")
        st.stop()

    with st.status("Running RAG pipeline…", expanded=True) as status:
        t0 = time.time()

        st.write(f"**Sequence:** {len(raw_seq)} chars → **{len(cleaned)} AAs** after cleaning")

        st.write("Embedding sequence with ESM-2…")
        query_vec = embed_sequence(cleaned, esm_model, esm_tokenizer)

        st.write(f"Retrieving {k_neighbors} nearest neighbors…")
        neighbors = retrieve_neighbors(query_vec, known_vecs, known_ids, multimodal_df, k=k_neighbors)

        st.write("Building cluster context…")
        context, cluster, confidence = build_rag_context(neighbors, cluster_kb)

        st.write("Generating LLM interpretation…")
        answer = llm_answer(context, question, neighbors, cluster, confidence,
                            cluster_kb, llm_model, llm_tokenizer, max_new_tokens=max_tokens)

        elapsed = time.time() - t0
        status.update(label=f"Done in {elapsed:.1f}s ✅", state="complete")

    # ── Results ----------------------------------------------------------------
    st.markdown("---")

    # Summary banner
    cluster_label = "NOISE / unclassified" if cluster == -1 else f"Cluster **{cluster}**"
    conf_color = "green" if confidence >= 0.67 else "orange"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Cluster", "NOISE" if cluster == -1 else f"Cluster {cluster}")
    col_b.metric("Confidence", f"{confidence:.0%}")
    col_c.metric("Top-1 Similarity", f"{neighbors[0]['similarity']:.4f}")

    # AI answer
    st.subheader("🤖 AI Interpretation")
    st.info(answer)

    # Nearest neighbors table
    st.subheader(f"🔎 Top-{k_neighbors} Nearest Neighbors")
    rows = []
    for n in neighbors:
        d = n["data"]
        rows.append({
            "Rank": n["rank"],
            "ID": n["id"],
            "Similarity": f"{n['similarity']:.4f}",
            "Cluster": int(d.get("super_cluster", -1)),
            "Protein Name": str(d.get("Protein names", "N/A"))[:80],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Cluster profile
    kb = cluster_kb.get(cluster, {})
    if kb:
        st.subheader("🧪 Cluster Knowledge Base Profile")
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown(f"**Cluster size:** {kb.get('size', '?')} proteins")
            st.markdown("**GO terms:**")
            for g in kb.get("go_terms", [])[:6]:
                st.markdown(f"  - {g}")
        with col_y:
            st.markdown("**Keywords:**")
            for kw in kb.get("keywords", [])[:10]:
                st.markdown(f"  - {kw}")

        if kb.get("sample_functions"):
            st.markdown("**Known functions in cluster:**")
            for fn in kb.get("sample_functions", [])[:2]:
                st.markdown(f"> {fn[:500]}")

    # Raw context expander
    with st.expander("📄 Raw RAG context sent to LLM"):
        st.code(context, language="text")
