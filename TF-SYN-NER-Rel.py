"""
MK News Topic Modeling Pipeline (FINANCE)
- Filter: finance/economy keywords
- Standard TF-IDF vs Custom TF-new (POS + Syntax + Facts + NER)
- Topic coherence (C_v) comparison across topic counts
- Numeric tokens removed: any lemma containing digits
"""

import os
import re
import logging
from collections import defaultdict, Counter

import pandas as pd
import pymorphy2
import matplotlib.pyplot as plt
from tqdm import tqdm

from razdel import tokenize, sentenize
from navec import Navec
from slovnet import Morph, Syntax, NER

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


CONFIG = {
    "data": {
        "filepath": "Новости_МК_full_финал_2.csv",
        "chunk_size": 50000,
        "rename_columns": {"Текст": "full_text", "Тема": "header_text"},
        "required_columns": ["full_text", "header_text"],
        "sample_size": None,
        "random_state": 42,
    },
    "models": {
        "batch_size": 4,
        "paths": {
            "navec": os.path.join(os.path.expanduser("~"), ".natasha", "navec",
                                 "navec_news_v1_1B_250K_300d_100q.tar"),
            "morph": os.path.join(os.path.expanduser("~"), ".natasha", "slovnet",
                                 "slovnet_morph_news_v1.tar"),
            "syntax": os.path.join(os.path.expanduser("~"), ".natasha", "slovnet",
                                  "slovnet_syntax_news_v1.tar"),
            "ner": os.path.join(os.path.expanduser("~"), ".natasha", "slovnet",
                               "slovnet_ner_news_v1.tar"),
        },
    },
    "filter": {
        "domain_name": "FINANCE",
        "keywords": {
            "экономика", "финансы", "бюджет", "налог", "налоги", "пошлина",
            "инфляция", "дефляция", "ввп", "дефицит", "профицит", "долг",
            "центробанк", "цб", "ключевая ставка", "ставка", "рефинансирование",
            "банк", "банки", "кредит", "ипотека", "депозит", "вклад",
            "биржа", "акции", "облигации", "дивиденд", "котировки", "индекс",
            "инвестиция", "инвестор", "фонд", "фондовый", "капитализация",
            "рынок", "ценные бумаги",
            "валюта", "курс", "рубль", "доллар", "евро", "юань",
            "нефть", "газ", "баррель",
            "прибыль", "выручка", "убыток", "банкротство",
            "сделка", "слияние", "поглощение",
        },
    },
    "preprocessing": {
        "stop_pos": {"PUNCT", "ADP", "CCONJ", "SCONJ", "PART", "PRON", "DET"},
        "general_stop_words": {
            "быть", "мочь", "стать", "являться", "так", "вот", "уже", "очень", "весь", "это",
            "который", "свой", "год", "человек", "сказать", "говорить", "также"
        },
        "remove_tokens_with_any_digits": True,
    },
    "tf_new": {
        "enabled": True,
        "use_ner": True,
        "ner_lambda": 1.5,
        "use_facts": True,
        "fact_lambda": 1.5,
        "pos_weights": {"in_title": 2.0, "in_first_sentence": 1.5, "default": 1.0},
        "syntax_weights": {"root": 2.0, "nsubj": 1.8, "obj": 1.5, "amod": 1.1, "default": 1.0},
    },
    "vectorizers": {
        "standard_tfidf": {"min_df": 10, "max_df": 0.6},
        "custom_idf": {"use_idf": True, "norm": None, "min_df": 10, "max_df": 0.6},
    },
    "lda": {
        "topic_min": 3,
        "topic_max": 10,           
        "random_state": 42,
        "topic_words_to_print": 10 
    },
    "coherence": {"metric": "c_v"},
    "plot": {
        "figsize": (12, 7),
        "title": "Topic Coherence (C_v) comparison (FINANCE)",
        "xlabel": "Number of topics (n_components)",
        "ylabel": "Coherence Score (C_v)",
        "label_custom": "Custom TF-new",
        "label_standard": "Standard TF-IDF",
    },
    "output": {
        "print_coherence_table": True, 
        "print_ner_stats": True,
    },
    "runtime": {
        "quiet_external_logs": True,
    }
}


# ----------------------- quiet external logs -----------------------
def setup_quiet_mode():
    if not CONFIG["runtime"].get("quiet_external_logs", True):
        return
    logging.basicConfig(level=logging.WARNING)
    for name in (
        "gensim", "gensim.corpora.dictionary", "gensim.models.coherencemodel",
        "pymorphy2", "pymorphy2.analyzer", "smart_open",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ----------------------- numeric filter -----------------------
_DIGIT_RE = re.compile(r"\d")


def should_skip_lemma(lemma: str) -> bool:
    if not lemma:
        return True
    return bool(CONFIG["preprocessing"]["remove_tokens_with_any_digits"] and _DIGIT_RE.search(lemma))


# ----------------------- load NLP -----------------------
def load_nlp_components():
    paths = CONFIG["models"]["paths"]
    bs = CONFIG["models"]["batch_size"]

    navec = Navec.load(paths["navec"])
    morph = Morph.load(paths["morph"], batch_size=bs)
    syntax = Syntax.load(paths["syntax"], batch_size=bs)
    ner = NER.load(paths["ner"], batch_size=bs)

    morph.navec(navec)
    syntax.navec(navec)
    ner.navec(navec)

    pymorph_parser = pymorphy2.MorphAnalyzer()
    return morph, syntax, ner, pymorph_parser


# ----------------------- NLP processing -----------------------
def process_text_final(text, morph, syntax, ner, pymorph_parser):
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return []

    tokens = list(tokenize(text))
    token_texts = [t.text for t in tokens]

    morph_markup = next(morph.map([token_texts]))
    syntax_markup = next(syntax.map([token_texts]))
    ner_markup = next(ner.map([text]))

    processed = []
    for tok, m_tok, s_tok in zip(tokens, morph_markup.tokens, syntax_markup.tokens):
        lemma = pymorph_parser.parse(tok.text)[0].normal_form
        processed.append({
            "text": tok.text,
            "start": tok.start,
            "stop": tok.stop,
            "pos": m_tok.pos,
            "lemma": lemma,
            "feats": m_tok.feats,
            "head_id": s_tok.head_id,
            "rel": s_tok.rel,
            "ner_type": None,
        })

    for span in ner_markup.spans:
        for t in processed:
            if t["start"] >= span.start and t["stop"] <= span.stop:
                t["ner_type"] = span.type

    return processed


def extract_facts_from_tokens(tokens):
    facts = []
    boundaries = [i for i, t in enumerate(tokens) if t.get("head_id") == "0"]
    start = 0

    for end in boundaries:
        sent = tokens[start:end + 1]
        root = tokens[end]
        root_idx = len(sent) - 1

        subj, obj = None, None
        for t in sent:
            if t.get("head_id") == str(root_idx + 1):
                if t.get("rel") == "nsubj":
                    subj = t
                elif t.get("rel") == "obj":
                    obj = t

        if subj and obj:
            facts.append({"subject": subj, "action": root, "object": obj})

        start = end + 1

    return facts


def calculate_tf_new(df, morph, syntax, ner, pymorph_parser):
    cfg = CONFIG["tf_new"]
    stop_pos = CONFIG["preprocessing"]["stop_pos"]
    stop_words = CONFIG["preprocessing"]["general_stop_words"]

    pos_w = cfg["pos_weights"]
    syn_w = cfg["syntax_weights"]

    use_facts = cfg["use_facts"]
    fact_lambda = cfg["fact_lambda"]
    use_ner = cfg["use_ner"]
    ner_lambda = cfg["ner_lambda"]

    tqdm.pandas(desc="TF_new")
    def row_tf_new(row):
        tokens = row["processed_tokens"]
        if not tokens:
            return {}

        header_text = str(row.get("header_text", ""))
        header_lemmas = {t["lemma"] for t in process_text_final(header_text, morph, syntax, ner, pymorph_parser)}

        sents = list(sentenize(row["full_text"]))
        first_span = (sents[0].start, sents[0].stop) if sents else None

        fact_lemmas = set()
        if use_facts:
            facts = extract_facts_from_tokens(tokens)
            fact_lemmas = {f[role]["lemma"] for f in facts for role in f}

        scores = defaultdict(float)

        for t in tokens:
            lemma = t["lemma"]
            if t["pos"] in stop_pos:
                continue
            if lemma in stop_words:
                continue
            if should_skip_lemma(lemma):
                continue

            if lemma in header_lemmas:
                w_pos = pos_w["in_title"]
            elif first_span and t["start"] >= first_span[0] and t["stop"] <= first_span[1]:
                w_pos = pos_w["in_first_sentence"]
            else:
                w_pos = pos_w["default"]

            w_syn = syn_w.get(t.get("rel", ""), syn_w["default"])
            w_fact = fact_lambda if (use_facts and lemma in fact_lemmas) else 1.0
            w_ner = ner_lambda if (use_ner and t.get("ner_type") is not None) else 1.0

            scores[lemma] += w_pos * w_syn * w_fact * w_ner

        return dict(scores)

    df["tf_new"] = df.progress_apply(row_tf_new, axis=1)
    return df


def get_topics_lists(model, feature_names, n_words):
    topics = []
    for weights in model.components_:
        idx = weights.argsort()[:-n_words - 1:-1]
        topics.append([feature_names[i] for i in idx])
    return topics


def compute_ner_stats(processed_docs):
    total = 0
    tagged = 0
    types = Counter()
    for doc in processed_docs:
        for t in doc:
            total += 1
            if t.get("ner_type") is not None:
                tagged += 1
                types[t["ner_type"]] += 1
    ratio = (tagged / total) if total else 0.0
    return total, tagged, ratio, types


def filter_domain(df):
    kws = CONFIG["filter"]["keywords"]
    pattern = "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True))
    search = (
        df["header_text"].fillna("").astype(str).str.lower()
        + " "
        + df["full_text"].fillna("").astype(str).str.lower()
    )
    mask = search.str.contains(pattern, na=False, regex=True)
    return df[mask].copy()


def main():
    setup_quiet_mode()

    print("Loading NLP models...")
    morph, syntax, ner, pymorph_parser = load_nlp_components()

    print("Loading dataset...")
    dcfg = CONFIG["data"]
    chunks = []
    reader = pd.read_csv(dcfg["filepath"], chunksize=dcfg["chunk_size"])
    for chunk in tqdm(reader, desc="Load CSV", dynamic_ncols=True):
        chunk.rename(columns=dcfg["rename_columns"], inplace=True)
        chunk.dropna(subset=["full_text"], inplace=True)
        if "header_text" not in chunk.columns:
            chunk["header_text"] = ""
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    missing = [c for c in dcfg["required_columns"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after renaming: {missing}")

    print(f"Filtering domain: {CONFIG['filter']['domain_name']} ...")
    df_domain = filter_domain(df)
    print(f"Domain articles: {len(df_domain)} (from {len(df)})")

    if dcfg["sample_size"] is not None and len(df_domain) > dcfg["sample_size"]:
        df_domain = df_domain.sample(n=dcfg["sample_size"], random_state=dcfg["random_state"]).copy()
        print(f"Using sample: {len(df_domain)}")

    print("NLP preprocessing...")
    processed_docs = [
        process_text_final(text, morph, syntax, ner, pymorph_parser)
        for text in tqdm(df_domain["full_text"], desc="NLP", dynamic_ncols=True)
    ]
    df_domain["processed_tokens"] = processed_docs

    if CONFIG["output"]["print_ner_stats"]:
        total, tagged, ratio, types = compute_ner_stats(processed_docs)
        top = types.most_common(3)
        print(f"NER tagged: {tagged}/{total} ({ratio*100:.2f}%), top: {top}")

    print("Building TF-IDF matrices...")
    stop_pos = CONFIG["preprocessing"]["stop_pos"]
    stop_words = CONFIG["preprocessing"]["general_stop_words"]

    corpus_lemmas = df_domain["processed_tokens"].apply(
        lambda toks: " ".join(
            [
                t["lemma"] for t in toks
                if t["pos"] not in stop_pos
                and t["lemma"] not in stop_words
                and not should_skip_lemma(t["lemma"])
            ]
        )
    )

    vstd = CONFIG["vectorizers"]["standard_tfidf"]
    tfidf_std = TfidfVectorizer(min_df=vstd["min_df"], max_df=vstd["max_df"])
    X_std = tfidf_std.fit_transform(corpus_lemmas)
    feat_std = tfidf_std.get_feature_names_out()

    df_domain = calculate_tf_new(df_domain, morph, syntax, ner, pymorph_parser)

    corpus_custom = df_domain["tf_new"].apply(lambda d: " ".join(d.keys()))
    vidf = CONFIG["vectorizers"]["custom_idf"]
    idf_vec = TfidfVectorizer(
        use_idf=vidf["use_idf"], norm=vidf["norm"],
        min_df=vidf["min_df"], max_df=vidf["max_df"],
    )
    idf_vec.fit(corpus_custom)
    idf_map = dict(zip(idf_vec.get_feature_names_out(), idf_vec.idf_))

    dv = DictVectorizer()
    X_tf = dv.fit_transform(df_domain["tf_new"])
    feat_custom = dv.get_feature_names_out()

    idf_weights = [idf_map.get(f, 1.0) for f in feat_custom]
    X_custom = X_tf.multiply(idf_weights)

    print(f"Matrix shapes: standard={X_std.shape}, custom={X_custom.shape}")

    print("Computing coherence...")
    lda_cfg = CONFIG["lda"]
    topic_range = range(lda_cfg["topic_min"], lda_cfg["topic_max"] + 1)

    texts = corpus_lemmas.apply(str.split).tolist()
    dictionary = Dictionary(texts)

    c_std, c_custom = [], []
    for k in tqdm(topic_range, desc="Coherence", dynamic_ncols=True):
        lda_s = LatentDirichletAllocation(n_components=k, random_state=lda_cfg["random_state"]).fit(X_std)
        topics_s = get_topics_lists(lda_s, feat_std, n_words=10)
        cm_s = CoherenceModel(topics=topics_s, texts=texts, dictionary=dictionary,
                              coherence=CONFIG["coherence"]["metric"])
        c_std.append(cm_s.get_coherence())

        lda_c = LatentDirichletAllocation(n_components=k, random_state=lda_cfg["random_state"]).fit(X_custom)
        topics_c = get_topics_lists(lda_c, feat_custom, n_words=10)
        cm_c = CoherenceModel(topics=topics_c, texts=texts, dictionary=dictionary,
                              coherence=CONFIG["coherence"]["metric"])
        c_custom.append(cm_c.get_coherence())

    results_df = pd.DataFrame({
        "Number of topics": list(topic_range),
        "Coherence (Custom)": c_custom,
        "Coherence (Standard)": c_std,
    })

    best_row = results_df.iloc[results_df["Coherence (Custom)"].idxmax()]
    best_k = int(best_row["Number of topics"])
    best_c = float(best_row["Coherence (Custom)"])
    best_s = float(best_row["Coherence (Standard)"])

    if CONFIG["output"]["print_coherence_table"]:
        print("\nCoherence table:")
        print(results_df.round(4))
    print(f"\nBest N (by Custom): {best_k} | Custom={best_c:.4f} | Standard={best_s:.4f}")

    print("\nTopics (Custom TF-new):")
    lda_custom_final = LatentDirichletAllocation(n_components=best_k, random_state=lda_cfg["random_state"]).fit(X_custom)
    topn = lda_cfg["topic_words_to_print"]
    for i, topic in enumerate(lda_custom_final.components_):
        idx = topic.argsort()[:-(topn + 1):-1]
        words = [feat_custom[j] for j in idx]
        print(f"Topic {i + 1}: {', '.join(words)}")

    print("\nTopics (Standard TF-IDF):")
    lda_std_final = LatentDirichletAllocation(n_components=best_k, random_state=lda_cfg["random_state"]).fit(X_std)
    for i, topic in enumerate(lda_std_final.components_):
        idx = topic.argsort()[:-(topn + 1):-1]
        words = [feat_std[j] for j in idx]
        print(f"Topic {i + 1}: {', '.join(words)}")

    # Plot
    plot = CONFIG["plot"]
    plt.figure(figsize=plot["figsize"])
    plt.plot(list(topic_range), c_custom, "bo-", label=plot["label_custom"])
    plt.plot(list(topic_range), c_std, "r--o", label=plot["label_standard"])
    plt.title(plot["title"], fontsize=16)
    plt.xlabel(plot["xlabel"])
    plt.ylabel(plot["ylabel"])
    plt.xticks(list(topic_range))
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

