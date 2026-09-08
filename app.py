"""
Streamlit application for the
Arabic Character-Level Language Model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


# PROJECT PATH

PROJECT_ROOT = Path(
    __file__
).resolve().parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# IMPORTS

from src.preprocessing import (
    load_text_file,
    clean_dataset,
)

from src.tokenizer import (
    EnhancedTokenizer,
)

from src.ngram import (
    CharacterNGramModel,
)

from src.generator import (
    generate_text,
)

from src.evaluation import (
    evaluate_model,
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title=(
        "Arabic Character-Level "
        "Language Model"
    ),
    page_icon="🧠",
    layout="wide",
)


# CUSTOM CSS

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .arabic-output {
        direction: rtl;
        text-align: right;
        font-size: 28px;
        line-height: 2;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
        margin-top: 15px;
    }

    .metric-description {
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# TITLE

st.markdown(
    """
    <div class="main-title">
        🧠 Arabic Character-Level Language Model
    </div>

    <div class="subtitle">
        Building an Arabic language model from scratch
        using character-level N-grams.
    </div>
    """,
    unsafe_allow_html=True,
)


# SIDEBAR

st.sidebar.title(
    "⚙️ Model Settings"
)

n = st.sidebar.selectbox(
    "N-gram model",
    options=[2, 3, 4],
    format_func=lambda x: (
        "Bigram (2-gram)"
        if x == 2
        else
        "Trigram (3-gram)"
        if x == 3
        else
        "4-gram"
    ),
)

sampling_mode = st.sidebar.selectbox(
    "Generation mode",
    options=[
        "random",
        "greedy",
    ],
)

n_tokens = st.sidebar.slider(
    "Characters to generate",
    min_value=1,
    max_value=200,
    value=50,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42,
)


# DATASET

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arabic_sample.txt"
)


@st.cache_data
def load_dataset():

    if DEFAULT_DATASET.exists():

        dataset = load_text_file(
            DEFAULT_DATASET
        )

        return clean_dataset(
            dataset
        )

    # -----------------------------------------------------
    # Fallback demo dataset
    # -----------------------------------------------------

    return [
        "الشمس طلعت في الصباح",
        "ذهب الطفل إلى المدرسة",
        "كان الجو جميلا اليوم",
        "اللغة العربية لغة جميلة",
        "يحب الطالب قراءة الكتب",
        "جلس الرجل تحت الشجرة",
        "في الصباح ذهبنا إلى السوق",
        "عاد الطفل إلى البيت",
        "السماء صافية اليوم",
        "المدينة كبيرة وجميلة",
    ]


dataset = load_dataset()


# TRAIN MODEL

@st.cache_resource
def train_model(
    dataset_tuple,
    n,
):

    dataset = list(
        dataset_tuple
    )

    tokenizer = EnhancedTokenizer(
        dataset
    )

    model = CharacterNGramModel(
        n=n
    )

    model.fit(
        dataset,
        tokenizer,
    )

    return tokenizer, model


tokenizer, model = train_model(
    tuple(dataset),
    n,
)


# TOP METRICS

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Dataset Samples",
        len(dataset),
    )

with col2:

    st.metric(
        "Vocabulary",
        tokenizer.vocabulary_size,
    )

with col3:

    st.metric(
        "N-gram Contexts",
        len(model),
    )

with col4:

    st.metric(
        "N",
        n,
    )


# TABS

tab1, tab2, tab3 = st.tabs(
    [
        "✍️ Text Generation",
        "📊 Model Information",
        "📈 Evaluation",
    ]
)


# TAB 1 — GENERATION

with tab1:

    st.header(
        "Generate Arabic Text"
    )

    st.write(
        """
        Enter an Arabic starting prompt and let the
        character-level N-gram model continue the text.
        """
    )

    prompt = st.text_input(
        "Starting prompt",
        value="يوم واحد",
    )

    if st.button(
        "🚀 Generate Text",
        type="primary",
    ):

        if not prompt.strip():

            st.warning(
                "Please enter a starting prompt."
            )

        else:

            generated_text = (
                generate_text(
                    start_prompt=prompt,
                    n_tokens=n_tokens,
                    model=model,
                    tokenizer=tokenizer,
                    sampling_mode=(
                        sampling_mode
                    ),
                    seed=int(
                        random_seed
                    ),
                )
            )

            st.subheader(
                "Generated Text"
            )

            st.markdown(
                f"""
                <div class="arabic-output">
                    {generated_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"Model: {n}-gram | "
                f"Mode: {sampling_mode} | "
                f"Generated characters: "
                f"{n_tokens}"
            )


# TAB 2 — MODEL INFORMATION

with tab2:

    st.header(
        "Model Information"
    )

    st.write(
        f"""
        The current model is a **character-level
        {n}-gram language model**.
        """
    )

    if n == 2:

        st.code(
            "P(character | previous_character)"
        )

    elif n == 3:

        st.code(
            "P(character | previous_2_characters)"
        )

    else:

        st.code(
            "P(character | previous_3_characters)"
        )

    st.subheader(
        "Example Vocabulary"
    )

    vocabulary = tokenizer.vocabulary

    preview = vocabulary[:100]

    st.write(
        " ".join(
            f"`{token}`"
            for token in preview
        )
    )

    st.subheader(
        "Example Contexts"
    )

    contexts = model.get_contexts()

    if contexts:

        for context in contexts[:20]:

            probabilities = (
                model.get_next_token_probabilities(
                    context
                )
            )

            next_token = max(
                probabilities,
                key=probabilities.get,
            )

            probability = (
                probabilities[next_token]
            )

            st.write(
                f"`{context}` → "
                f"`{next_token}` "
                f"({probability:.2%})"
            )


# TAB 3 — EVALUATION

with tab3:

    st.header(
        "Model Evaluation"
    )

    st.write(
        """
        The model is evaluated using the same dataset
        used for demonstration. For a rigorous experiment,
        use the train/test evaluation pipeline from Week 3.
        """
    )

    metrics = evaluate_model(
        model,
        dataset,
        tokenizer,
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        st.metric(
            "Cross Entropy",
            f"{metrics['cross_entropy']:.4f}",
        )

        st.caption(
            "Lower is better."
        )

    with col2:

        st.metric(
            "Perplexity",
            f"{metrics['perplexity']:.4f}",
        )

        st.caption(
            "Lower is better."
        )

    st.divider()

    st.subheader(
        "How to interpret the metrics"
    )

    st.markdown(
        """
        **Cross Entropy**

        Measures how well the model predicts the correct
        next character.

        Lower cross entropy means better predictions.

        **Perplexity**

        Perplexity can be interpreted as how uncertain
        the model is when predicting the next character.

        Lower perplexity means the model is less uncertain.
        """
    )


# FOOTER

st.divider()

st.caption(
    "Arabic Character-Level Language Model "
    "— Built from scratch with Python and Streamlit"
)