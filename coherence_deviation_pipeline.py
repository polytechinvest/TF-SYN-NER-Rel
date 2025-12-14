"""
TF-SYN-NER-Rel — Topic Coherence Comparison (C_v)

This script builds two document-term matrices from the same corpus:
1) Standard TF-IDF (baseline)
2) Custom TF-new (TF-SYN-NER-Rel), where term frequency is re-weighted using
   positional, syntactic and factual cues, and then multiplied by IDF.

For each number of topics in a given range and for multiple random seeds, the script:
- fits an LDA model (scikit-learn) for each representation,
- extracts top-N words per topic,
- computes topic coherence (c_v) with gensim,
- aggregates mean ± std across seeds,
- plots the comparison (error bars).

Default settings reproduce the original experiment setup used to generate Cv_Std.png.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy2
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from navec import Navec
from razdel import sentenize, tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from slovnet import Morph, NER, Syntax
from tqdm import tqdm


STOP_POS = {"PUNCT", "ADP", "CCONJ", "SCONJ", "PART", "PRON", "DET"}


def _safe_csv_field_size_limit() -> None:
    """Increase CSV field size limit where possible (helps with very long text fields)."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(int(2147483647 / 2))


def load_nlp_components(batch_size: int = 4) -> Tuple[Navec, Morph, Syntax, NER, pymorphy2.MorphAnalyzer]:
    """Load Natasha/Slovnet components from the default ~/.natasha folder."""
    home_dir = os.path.expanduser("~")
    navec_path = os.path.join(home_dir, ".natasha", "navec", "navec_news_v1_1B_250K_300d_100q.tar")
    morph_path = os.path.join(home_dir, ".natasha", "slovnet", "slovnet_morph_news_v1.tar")
    syntax_path = os.path.join(home_dir, ".natasha", "slovnet", "slovnet_syntax_news_v1.tar")
    ner_path = os.path.join(home_dir, ".natasha", "slovnet", "slovnet_ner_news_v1.tar")

    navec = Navec.load(navec_path)
    morph = Morph.load(morph_path, batch_size=batch_size)
    syntax = Syntax.load(syntax_path, batch_size=batch_size)
    ner = NER.load(ner_path, batch_size=batch_size)

    morph.navec(navec)
    syntax.navec(navec)
    ner.navec(navec)

    pymorph = pymorphy2.MorphAnalyzer()
    return navec, morph, syntax, ner, pymorph


def detect_text_and_header_columns(df_head: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Heuristically detect text and (optionally) header columns in a CSV."""
    possible_text_columns = ["Текст", "Рассказ", "full_text", "text", "Text"]
    possible_header_columns = ["Тема", "Заголовок", "header_text", "Header", "header"]

    text_column: Optional[str] = None
    for col in possible_text_columns:
        if col in df_head.columns:
            text_column = col
            break
    if text_column is None:
        # fallback: first non-empty object column
        for col in df_head.columns:
            if df_head[col].dtype == "object" and df_head[col].notna().any():
                text_column = col
                break
    if text_column is None:
        raise KeyError(f"Could not find a text column. Available columns: {list(df_head.columns)}")

    header_column: Optional[str] = None
    for col in possible_header_columns:
        if col in df_head.columns:
            header_column = col
            break

    return text_column, header_column


def process_text_final(
    text: str,
    *,
    morph: Morph,
    syntax: Syntax,
    ner: NER,
    pymorph: pymorphy2.MorphAnalyzer,
) -> List[Dict]:
    """Tokenize text, add morphology, syntax and NER tags, and lemmatize with pymorphy2."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []

    tokens = list(tokenize(text))
    token_texts = [t.text for t in tokens]

    morph_markup = next(morph.map([token_texts]))
    syntax_markup = next(syntax.map([token_texts]))
    ner_markup = next(ner.map([text]))

    processed: List[Dict] = []
    for token, morph_token, syntax_token in zip(tokens, morph_markup.tokens, syntax_markup.tokens):
        lemma = pymorph.parse(token.text)[0].normal_form
        processed.append(
            {
                "text": token.text,
                "start": token.start,
                "stop": token.stop,
                "pos": morph_token.pos,
                "lemma": lemma,
                "feats": morph_token.feats,
                "head_id": syntax_token.head_id,
                "rel": syntax_token.rel,
                "ner_type": None,
            }
        )

    for span in ner_markup.spans:
        for tok in processed:
            if tok["start"] >= span.start and tok["stop"] <= span.stop:
                tok["ner_type"] = span.type

    return processed


def extract_facts_from_tokens(processed_tokens: Sequence[Dict]) -> List[Dict]:
    """Extract simple subject-action-object facts from Slovnet dependency output."""
    facts: List[Dict] = []
    # sentence boundary heuristic: head_id == '0' marks root token of a sentence
    sentence_roots = [i for i, tok in enumerate(processed_tokens) if tok.get("head_id") == "0"]

    start_idx = 0
    for end_idx in sentence_roots:
        sentence_tokens = processed_tokens[start_idx : end_idx + 1]
        root_token = processed_tokens[end_idx]
        root_idx_in_sentence = len(sentence_tokens) - 1

        subject = None
        obj = None
        for tok in sentence_tokens:
            if tok.get("head_id") == str(root_idx_in_sentence + 1):
                if tok.get("rel") == "nsubj":
                    subject = tok
                elif tok.get("rel") == "obj":
                    obj = tok

        if subject and obj:
            facts.append({"subject": subject, "action": root_token, "object": obj})

        start_idx = end_idx + 1

    return facts


def calculate_tf_new(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TF-new weights per document and store them in df['tf_new'] as a dict(lemma -> score)."""
    tqdm.pandas(desc="Calculating TF_new")

    def tf_new_for_row(row: pd.Series) -> Dict[str, float]:
        processed_tokens: List[Dict] = row["processed_tokens"]
        if not processed_tokens:
            return {}

        sents = list(sentenize(row["full_text"]))
        first_sent_span = (sents[0].start, sents[0].stop) if sents else None

        facts = extract_facts_from_tokens(processed_tokens)
        fact_lemmas = {fact[role]["lemma"] for fact in facts for role in fact}

        scores: defaultdict[str, float] = defaultdict(float)
        for tok in processed_tokens:
            if tok.get("pos") in STOP_POS:
                continue

            pos_w = (
                1.5
                if first_sent_span
                and tok["start"] >= first_sent_span[0]
                and tok["stop"] <= first_sent_span[1]
                else 1.0
            )

            syn_w_map = {"root": 2.0, "nsubj": 1.8, "obj": 1.5, "amod": 1.1}
            syn_w = syn_w_map.get(tok.get("rel", ""), 1.0)

            fact_w = 1.5 if tok["lemma"] in fact_lemmas else 1.0

            scores[tok["lemma"]] += pos_w * syn_w * fact_w

        return dict(scores)

    df["tf_new"] = df.progress_apply(tf_new_for_row, axis=1)
    return df


def get_topics_lists(model: LatentDirichletAllocation, feature_names: Sequence[str], n_words: int = 10) -> List[List[str]]:
    """Extract top-N words per topic from a fitted scikit-learn LDA model."""
    topics: List[List[str]] = []
    for topic_weights in model.components_:
        top_idx = topic_weights.argsort()[:-n_words - 1 : -1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def build_matrices(
    stories_df: pd.DataFrame,
    *,
    min_df: int,
    max_df: float,
) -> Tuple:
    """Build standard TF-IDF matrix and custom TF-new*IDF matrix."""
    # baseline (standard TF-IDF)
    corpus_lemmas = stories_df["processed_tokens"].apply(
        lambda toks: " ".join([t["lemma"] for t in toks if t.get("pos") not in STOP_POS])
    )
    tfidf_vectorizer_std = TfidfVectorizer(min_df=min_df, max_df=max_df)
    matrix_std = tfidf_vectorizer_std.fit_transform(corpus_lemmas)
    feature_names_std = tfidf_vectorizer_std.get_feature_names_out()

    # custom (TF-new * IDF)
    stories_df = calculate_tf_new(stories_df)

    corpus_for_idf_custom = stories_df["tf_new"].apply(lambda d: " ".join(d.keys()))
    idf_vectorizer_custom = TfidfVectorizer(use_idf=True, norm=None, min_df=min_df, max_df=max_df)
    idf_vectorizer_custom.fit(corpus_for_idf_custom)
    idf_map = dict(zip(idf_vectorizer_custom.get_feature_names_out(), idf_vectorizer_custom.idf_))

    dict_vectorizer = DictVectorizer()
    tf_new_matrix = dict_vectorizer.fit_transform(stories_df["tf_new"])
    feature_names_custom = dict_vectorizer.get_feature_names_out()
    idf_vector = np.array([idf_map.get(feature, 1.0) for feature in feature_names_custom])
    matrix_custom = tf_new_matrix.multiply(idf_vector)

    texts_for_coherence = corpus_lemmas.apply(str.split).tolist()
    dictionary = Dictionary(texts_for_coherence)

    return (
        matrix_std,
        feature_names_std,
        matrix_custom,
        feature_names_custom,
        texts_for_coherence,
        dictionary,
    )


def compute_coherence_curve(
    *,
    matrix,
    feature_names: Sequence[str],
    texts_for_coherence: List[List[str]],
    dictionary: Dictionary,
    topic_range: Sequence[int],
    seeds: Sequence[int],
) -> Dict[int, List[float]]:
    """Compute coherence scores for each topic count across multiple seeds."""
    results: Dict[int, List[float]] = {n_topics: [] for n_topics in topic_range}

    for seed_idx, seed in enumerate(seeds, 1):
        for n_topics in tqdm(topic_range, desc=f"Seed {seed} ({seed_idx}/{len(seeds)})"):
            lda = LatentDirichletAllocation(n_components=int(n_topics), random_state=int(seed))
            lda.fit(matrix)

            topics = get_topics_lists(lda, feature_names)
            cm = CoherenceModel(
                topics=topics,
                texts=texts_for_coherence,
                dictionary=dictionary,
                coherence="c_v",
            )
            results[int(n_topics)].append(float(cm.get_coherence()))

    return results


def plot_comparison(
    topic_range: Sequence[int],
    mean_custom: Sequence[float],
    std_custom: Sequence[float],
    mean_std: Sequence[float],
    std_std: Sequence[float],
    *,
    out_path: Optional[str],
    show: bool,
) -> None:
    """Plot the main comparison figure (the 'first graph' / Cv_Std.png)."""
    plt.figure(figsize=(12, 7))

    plt.errorbar(
        topic_range,
        mean_custom,
        yerr=std_custom,
        fmt="bo-",
        capsize=5,
        capthick=2,
        label="Custom TF-new",
        markersize=8,
        linewidth=2,
        alpha=0.8,
    )
    plt.errorbar(
        topic_range,
        mean_std,
        yerr=std_std,
        fmt="r--o",
        capsize=5,
        capthick=2,
        label="Standard TF-IDF",
        markersize=8,
        linewidth=2,
        alpha=0.8,
    )

    plt.title("Topic Coherence Comparison (C_v)\nMean ± 1 Standard Deviation", fontsize=16, fontweight="bold")
    plt.xlabel("Number of Topics (n_components)", fontsize=13)
    plt.ylabel("Coherence Score (C_v)", fontsize=13)
    plt.xticks(list(topic_range))
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-SYN-NER-Rel topic coherence comparison (C_v).")
    parser.add_argument("--input", default="Новости_МК_full_финал_2.csv", help="Path to input CSV file.")
    parser.add_argument("--sample-size", type=int, default=15000, help="Number of documents to sample.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling rows.")
    parser.add_argument("--min-topics", type=int, default=3, help="Minimum number of topics.")
    parser.add_argument("--max-topics", type=int, default=15, help="Maximum number of topics (inclusive).")
    parser.add_argument("--n-seeds", type=int, default=10, help="How many seeds to use from --seeds list.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066],
        help="Random seeds for LDA random_state. The first --n-seeds will be used.",
    )
    parser.add_argument("--min-df", type=int, default=10, help="min_df for TF-IDF vectorizers.")
    parser.add_argument("--max-df", type=float, default=0.6, help="max_df for TF-IDF vectorizers.")
    parser.add_argument("--batch-size", type=int, default=4, help="batch_size for Slovnet models.")
    parser.add_argument("--plot-path", default="Cv_Std.png", help="Where to save the main plot (set empty to disable).")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (useful in headless runs).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    _safe_csv_field_size_limit()

    print("Loading NLP models...")
    try:
        _, morph, syntax, ner, pymorph = load_nlp_components(batch_size=args.batch_size)
    except Exception as e:
        print(f"[CRITICAL] Failed to load NLP models from ~/.natasha: {e}")
        return 2

    print(f"Loading data: {args.input}")
    try:
        df_head = pd.read_csv(args.input, nrows=5)
        text_col, header_col = detect_text_and_header_columns(df_head)

        stories_df = pd.read_csv(args.input)
        stories_df = stories_df.rename(columns={text_col: "full_text"})
        if header_col and header_col != text_col and header_col in stories_df.columns:
            stories_df = stories_df.rename(columns={header_col: "header_text"})

        before = len(stories_df)
        stories_df = stories_df.dropna(subset=["full_text"])
        after = len(stories_df)
        if after != before:
            print(f"Removed {before - after} empty rows. Remaining: {after}")

        if after > args.sample_size:
            rng = np.random.RandomState(args.sample_seed)
            sample_idx = rng.choice(after, args.sample_size, replace=False)
            stories_df = stories_df.iloc[sample_idx].reset_index(drop=True)
            print(f"Sampled {args.sample_size} rows (seed={args.sample_seed}).")
        else:
            print(f"Using all {after} rows (<= sample-size).")

    except FileNotFoundError:
        print(f"[CRITICAL] File not found: {args.input}")
        return 2
    except Exception as e:
        print(f"[CRITICAL] Failed to load/prepare data: {e}")
        return 2

    print("Running NLP preprocessing...")
    tqdm.pandas(desc="NLP pipeline")
    stories_df["processed_tokens"] = [
        process_text_final(text, morph=morph, syntax=syntax, ner=ner, pymorph=pymorph)
        for text in tqdm(stories_df["full_text"], desc="NLP pipeline")
    ]

    print("Building matrices...")
    (
        matrix_std,
        feature_names_std,
        matrix_custom,
        feature_names_custom,
        texts_for_coherence,
        dictionary,
    ) = build_matrices(stories_df, min_df=args.min_df, max_df=args.max_df)

    topic_range = list(range(int(args.min_topics), int(args.max_topics) + 1))
    seeds = list(args.seeds)[: int(args.n_seeds)]

    print(f"Computing coherence across seeds ({len(seeds)}) and topic counts ({topic_range[0]}..{topic_range[-1]})...")

    results_std = compute_coherence_curve(
        matrix=matrix_std,
        feature_names=feature_names_std,
        texts_for_coherence=texts_for_coherence,
        dictionary=dictionary,
        topic_range=topic_range,
        seeds=seeds,
    )
    results_custom = compute_coherence_curve(
        matrix=matrix_custom,
        feature_names=feature_names_custom,
        texts_for_coherence=texts_for_coherence,
        dictionary=dictionary,
        topic_range=topic_range,
        seeds=seeds,
    )

    mean_std = [float(np.mean(results_std[n])) for n in topic_range]
    std_std = [float(np.std(results_std[n])) for n in topic_range]
    mean_custom = [float(np.mean(results_custom[n])) for n in topic_range]
    std_custom = [float(np.std(results_custom[n])) for n in topic_range]

    results_df = pd.DataFrame(
        {
            "Number of Topics": topic_range,
            "Custom: Mean ± 1 Std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_custom, std_custom)],
            "Standard: Mean ± 1 Std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_std, std_std)],
        }
    )

    print("\n" + "=" * 80)
    print("FINAL TABLE: Mean C_v ± 1 Standard Deviation")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    plot_path = (args.plot_path or "").strip()
    plot_out = plot_path if plot_path else None
    plot_comparison(
        topic_range,
        mean_custom,
        std_custom,
        mean_std,
        std_std,
        out_path=plot_out,
        show=not args.no_show,
    )
    if plot_out:
        print(f"\n[OK] Saved plot to: {plot_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
