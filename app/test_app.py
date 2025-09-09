
# -*- coding: utf-8 -*-
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional
import streamlit.components.v1 as components

# --- CONFIG: CSV paths injected by the notebook ---
DEFAULT_TOPIC_SUMMARY = Path(r"C:\Users\linna\OneDrive\Documents\Python_Dev\topic-modeling\outputs\bertopic_topic_summary.csv")
DEFAULT_COMMENTS_DF = Path(r"C:\Users\linna\OneDrive\Documents\Python_Dev\topic-modeling\outputs\comments_with_bertopic.csv")

# Basic page config
st.set_page_config(page_title="Topic Explorer", layout="wide", initial_sidebar_state="expanded")

@st.cache_data(show_spinner=False)
def load_csv_fallback(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

topic_summary = load_csv_fallback(DEFAULT_TOPIC_SUMMARY)
comments_df = load_csv_fallback(DEFAULT_COMMENTS_DF)

if topic_summary is None or comments_df is None:
    st.warning("Topic summary or comments CSV not found. Put the files in the expected paths and reload.")
    st.info(f"Expected: topic_summary={DEFAULT_TOPIC_SUMMARY}, comments={DEFAULT_COMMENTS_DF}")
    st.stop()

# convenience column names (prefer display columns if present)
# 'display columns' are filtered after topics are generated to remove meaningless words -- LK has yet to build this in
TOP_WORDS_COL = "top_words_display" if "top_words_display" in topic_summary.columns else "top_words"
SAMPLE_COMMENTS_COL = "sample_comments_display" if "sample_comments_display" in topic_summary.columns else "sample_comments"
TOPIC_NUM_COL = "topic_num" if "topic_num" in topic_summary.columns else ("topic" if "topic" in topic_summary.columns else None)
if TOPIC_NUM_COL is not None:
    topic_summary[TOPIC_NUM_COL] = topic_summary[TOPIC_NUM_COL].astype(int)

# detect dominant-topic column in comments (make app amenable to top2vec or bert)
DOM_COL_CANDIDATES = ["bertopic_dominant_topic", "top2vec_dominant_topic", "dominant_topic", "topic", "topic_num"]
dom_col = next((c for c in DOM_COL_CANDIDATES if c in comments_df.columns), None)
if dom_col is None:
    dom_col = "bertopic_dominant_topic"
    comments_df[dom_col] = -1
comments_df[dom_col] = comments_df[dom_col].fillna(-1).astype(int)

# optional emotion column
EMOTION_COL = "top_emotion" if "top_emotion" in comments_df.columns else None

# map topic labels; keep -1 as Noise
topic_label_map = {}
if TOPIC_NUM_COL is not None and TOP_WORDS_COL in topic_summary.columns:
    topic_label_map = dict(zip(topic_summary[TOPIC_NUM_COL].astype(int), topic_summary[TOP_WORDS_COL].astype(str)))
topic_label_map[-1] = "Noise"

# --- Filtering and Display ---
# 1) docket (top)
docket_col = "docket_id" if "docket_id" in comments_df.columns else None
if docket_col:
    docket_choices = ["(All)"] + sorted(comments_df[docket_col].dropna().astype(str).unique().tolist())
else:
    docket_choices = ["(All)"]
chosen_docket = st.sidebar.selectbox("Docket", options=docket_choices, index=0)

# 2) search comments (keyword)
keyword = st.sidebar.text_input("Search comments (keyword)", value="")

# Apply docket + search early to compute topic choices relevant to the query
df_initial = comments_df.copy()
if chosen_docket and chosen_docket != "(All)" and docket_col:
    df_initial = df_initial[df_initial[docket_col].astype(str) == chosen_docket]
if keyword:
    kw = keyword.lower()
    df_for_topics = df_initial[df_initial.apply(lambda r: kw in str(r.get("comment_text","")).lower() or kw in str(r.get("sample_comments","")).lower(), axis=1)]
else:
    df_for_topics = df_initial

# 3) topics multi-select (choices come from docket+search filtered rows)
all_topics = sorted(set(df_for_topics[dom_col].unique().tolist()))
topic_choices = ["All"] + [str(t) for t in all_topics if t != -1] + (["Noise"] if -1 in all_topics else [])
selected_topics = st.sidebar.multiselect("Topics (multi-select)", options=topic_choices, default=["All"])

# 4) emotion selector
if EMOTION_COL:
    emotion_choices = ["All"] + sorted(df_initial[EMOTION_COL].dropna().unique().tolist())
    selected_emotion = st.sidebar.selectbox("Emotion", emotion_choices, index=0)
else:
    selected_emotion = "All"

# 5) min topic size slider
min_topic_size_max = int(topic_summary["size"].max()) if "size" in topic_summary.columns else 100
min_topic_size = st.sidebar.slider("Min topic size (filter topic list)", min_value=0, max_value=min_topic_size_max, value=0, step=1)

# 6) show only 'noise' comments
show_only_noise = st.sidebar.checkbox("Show only -1 (noise) docs", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Display options")
show_sample_in_cards = st.sidebar.checkbox("Show sample text in topic cards", value=True)
cards_per_row = st.sidebar.selectbox("Cards per row", options=[1,2,3,4], index=2)

# --- Apply filters in the same order (start from docket-filtered df) ---
filtered = comments_df.copy()
if chosen_docket and chosen_docket != "(All)" and docket_col:
    filtered = filtered[filtered[docket_col].astype(str) == chosen_docket]

# keyword search
if keyword:
    kw = keyword.lower()
    filtered = filtered[filtered.apply(lambda r: kw in str(r.get("comment_text","")).lower() or kw in str(r.get("sample_comments","")).lower(), axis=1)]

# topic selection (honor "Noise" sentinel)
if not ("All" in selected_topics or len(selected_topics) == 0):
    sel = []
    for t in selected_topics:
        if str(t).lower() == "noise":
            sel.append(-1)
        else:
            try:
                sel.append(int(t))
            except Exception:
                pass
    if sel:
        filtered = filtered[filtered[dom_col].isin(sel)]

# emotion filter
if EMOTION_COL and selected_emotion != "All":
    filtered = filtered[filtered[EMOTION_COL] == selected_emotion]

# show only noise
if show_only_noise:
    filtered = filtered[filtered[dom_col] == -1]

# compute large_topics set if min_topic_size used (based on global topic_summary)
if "size" in topic_summary.columns and min_topic_size > 0:
    large_topics = set(topic_summary[topic_summary["size"].astype(int) >= min_topic_size][TOPIC_NUM_COL].astype(int).tolist())
else:
    large_topics = None

# --- Update title and browser tab with chosen docket ---
visible_title = f"Topic Explorer — {chosen_docket}" if chosen_docket and chosen_docket != "(All)" else "Topic Explorer"
st.title(visible_title)
components.html(f"<script>document.title = \"{visible_title.replace('\"','\\\"')}\";</script>", height=0)

# --- summaries ---
k1, k2, k3 = st.columns([1,1,2])
k1.metric("Filtered comments", f"{len(filtered):,}")
k2.metric("Topics represented", f"{filtered[dom_col].nunique()}")
top_em = filtered[EMOTION_COL].value_counts().idxmax() if EMOTION_COL and not filtered[EMOTION_COL].empty else "N/A"
k3.metric("Top emotion", top_em)

st.markdown("---")

# --- Charts: topic sizes and emotion distribution (responds to filtering) ---
c1, c2 = st.columns([2,1])

with c1:
    st.subheader("Topic sizes")
    topic_counts = filtered[dom_col].value_counts().reset_index()
    topic_counts.columns = ["topic","count"]
    topic_counts["label"] = topic_counts["topic"].map(lambda t: topic_label_map.get(int(t), f"Topic {t}"))
    if not topic_counts.empty:
        fig = px.bar(topic_counts.sort_values("count"), x="count", y="label", orientation="h", text="count", height=420)
        fig.update_layout(xaxis_title="Number of documents", yaxis_title="Topic", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No topics to display (check filters).")

with c2:
    st.subheader("Emotion distribution")
    if EMOTION_COL:
        ec = filtered[EMOTION_COL].value_counts().reset_index()
        ec.columns = ["emotion","count"]
        if not ec.empty:
            fig2 = px.pie(ec, names="emotion", values="count")
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No emotion data in filtered set.")
    else:
        st.info("No emotion column available.")

st.markdown("---")

# --- Topic previews (show topic number + count in filtered set; green top-words subhead) ---
st.subheader("Topic previews")
display_topics = topic_summary.copy()

# restrict to topics that appear in the docket/search-filtered set
topics_in_filtered = set(filtered[dom_col].unique().tolist())
display_topics = display_topics[display_topics[TOPIC_NUM_COL].astype(int).isin(topics_in_filtered)]

# apply user topic picks (if not "All")
if not ("All" in selected_topics or len(selected_topics) == 0):
    sel_topics = set()
    for t in selected_topics:
        if str(t).lower() == "noise":
            sel_topics.add(-1)
        else:
            try:
                sel_topics.add(int(t))
            except Exception:
                pass
    display_topics = display_topics[display_topics[TOPIC_NUM_COL].astype(int).isin(sel_topics)]

# apply min topic size (global) if requested
if large_topics is not None:
    display_topics = display_topics[display_topics[TOPIC_NUM_COL].astype(int).isin(large_topics)]

# order and limit
if "size" in display_topics.columns:
    display_topics = display_topics.sort_values("size", ascending=False)
elif TOPIC_NUM_COL:
    display_topics = display_topics.sort_values(TOPIC_NUM_COL)

display_topics = display_topics.head(30).reset_index(drop=True)

# prepare per-topic doc counts from the filtered set
topic_counts_map = filtered[dom_col].value_counts().to_dict()

if display_topics.empty:
    st.info("No topic previews match filters.")
else:
    cols = st.columns(cards_per_row)
    for i, row in display_topics.iterrows():
        col = cols[i % cards_per_row]
        with col:
            tnum = int(row[TOPIC_NUM_COL]) if TOPIC_NUM_COL else i
            # count from filtered set (reflects current filters)
            cnt = int(topic_counts_map.get(tnum, 0))
            header = f"Topic {tnum} — {cnt} docs"
            # show header (topic number + count)
            st.markdown(f"### {header}")
            # green subhead with top words
            top_words = row.get(TOP_WORDS_COL, row.get("top_words", ""))
            if top_words:
                st.markdown(f"<div style='color:green;margin-bottom:6px'>{top_words}</div>", unsafe_allow_html=True)
            if show_sample_in_cards:
                sample = row.get(SAMPLE_COMMENTS_COL, row.get("sample_comments", ""))
                if sample:
                    st.caption(sample[:300] + ("..." if len(sample) > 300 else ""))
            if st.button(f"View comments (topic {tnum})", key=f"view_{tnum}"):
                sub = filtered[filtered[dom_col] == tnum]
                if sub.empty:
                    st.info("No comments for this topic (in current filters).")
                else:
                    st.write(f"Showing {len(sub)} comments for topic {tnum}")
                    st.dataframe(sub[["comment_id","comment_text"]].head(200), use_container_width=True)

st.markdown("---")

# --- View/download comments ---
st.subheader("Browse filtered comments")
preview_cols = ["comment_id","comment_text", dom_col]
if EMOTION_COL:
    preview_cols += [EMOTION_COL, "top_emotion_score"] if "top_emotion_score" in comments_df.columns else [EMOTION_COL]
available_cols = [c for c in preview_cols if c in filtered.columns]
st.dataframe(filtered[available_cols].head(500), height=400)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered comments", csv_bytes, file_name="filtered_comments.csv", mime="text/csv")

st.markdown("---")
st.caption("Use the controls on the left to filter docket, topics, emotions, and to search comments. Click topic cards to view related comments.")
