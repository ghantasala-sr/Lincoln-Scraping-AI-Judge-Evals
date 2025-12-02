import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from glob import glob

# Page Config
st.set_page_config(
    page_title="Memory Machines Dashboard",
    page_icon="🏛️",
    layout="wide"
)

# Title
st.title("🏛️ Memory Machines: Historical Event Validation")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", [
    "Project Overview",
    "Data Inspection",
    "Event Extraction",
    "Evaluation Results",
    "Validation Metrics"
])

# --- Helper Functions ---
@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def get_raw_files():
    files = glob("data/raw/*.txt") + glob("data/raw/*.json") + glob("data/normalized/*.json")
    return sorted(files)

# --- 1. Project Overview ---
if page == "Project Overview":
    st.header("Project Overview")
    st.markdown("""
    This dashboard visualizes the **Memory Machines** pipeline, which extracts and evaluates historical events 
    from primary and secondary sources (Project Gutenberg & Library of Congress).
    
    **Pipeline Stages:**
    1.  **Scraping**: Fetching raw text/JSON.
    2.  **Extraction**: Identifying key events using Hybrid Retrieval (Keywords + Embeddings).
    3.  **Evaluation**: Using an LLM Judge to compare accounts for consistency.
    4.  **Validation**: Statistical tests (Kappa, Self-Consistency) to ensure reliability.
    """)

# --- 2. Data Inspection ---
elif page == "Data Inspection":
    st.header("Data Inspection (Raw & Normalized)")
    
    files = get_raw_files()
    if not files:
        st.warning("No data files found.")
    else:
        selected_file = st.selectbox("Select File", files)
        
        if selected_file:
            st.markdown(f"**File:** `{selected_file}`")
            file_size = os.path.getsize(selected_file) / 1024
            st.caption(f"Size: {file_size:.2f} KB")
            
            with open(selected_file, 'r') as f:
                content = f.read()
                
            if selected_file.endswith(".json"):
                st.json(json.loads(content))
            else:
                st.text_area("Content", content, height=400)

# --- 3. Event Extraction ---
elif page == "Event Extraction":
    st.header("Extracted Events")
    
    events = load_json("data/extracted/events_hybrid.json")
    if not events:
        st.warning("No events found in data/extracted/events_hybrid.json")
    else:
        # Filter by Event Name
        event_names = sorted(list(set(e['event'] for e in events)))
        selected_event = st.selectbox("Filter by Event", ["All"] + event_names)
        
        filtered_events = events if selected_event == "All" else [e for e in events if e['event'] == selected_event]
        
        st.write(f"Showing {len(filtered_events)} extracted segments.")
        
        for i, event in enumerate(filtered_events):
            date_str = event.get('temporal_details', {}).get('date', 'Unknown Date')
            with st.expander(f"{event['event']} - {event['source_title']} ({date_str})"):
                st.markdown(f"**Author:** {event['author']}")
                st.markdown(f"**Source:** {event.get('source_url', 'Unknown Source')}")
                st.markdown("**Extracted Content:**")
                if 'claims' in event:
                    for claim in event['claims']:
                        st.write(f"- {claim}")
                elif 'text' in event:
                    st.info(event['text'])
                else:
                    st.warning("No content found.")
                    
                st.markdown("**Metadata:**")
                metadata = {
                    "source_id": event.get("source_id"),
                    "pipeline": event.get("pipeline"),
                    "tone": event.get("tone")
                }
                st.json(metadata)

# --- 4. Evaluation Results ---
elif page == "Evaluation Results":
    st.header("Evaluation Results")
    
    # Tab structure for different evaluation reports
    tab1, tab2 = st.tabs(["Detailed Consistency Report", "Full Consistency (Reliability)"])
    
    with tab1:
        report = load_json("data/evaluated/consistency_report.json")
        if not report:
            st.warning("No evaluation report found in data/evaluated/consistency_report.json")
        else:
            # Flatten the data
            flat_data = []
            for item in report:
                entry = {
                    "event": item.get("event"),
                    "source_a_author": item.get("source_a", {}).get("author"),
                    "source_b_author": item.get("source_b", {}).get("author"),
                    "consistency_score": item.get("evaluation", {}).get("consistency_score"),
                    "confidence": item.get("evaluation", {}).get("confidence"),
                    "reasoning": item.get("evaluation", {}).get("reasoning"),
                    "contradictions": item.get("evaluation", {}).get("contradictions", [])
                }
                flat_data.append(entry)
            
            df = pd.DataFrame(flat_data)
            
            # Histogram
            st.subheader("Consistency Score Distribution")
            if "consistency_score" in df.columns:
                fig = px.histogram(df, x="consistency_score", nbins=20, title="Score Distribution", labels={"consistency_score": "Score"})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Table
            st.subheader("Detailed Comparison")
            st.dataframe(df[["event", "source_a_author", "source_b_author", "consistency_score", "confidence"]])
            
            # Drill Down
            st.subheader("Drill Down")
            selected_row = st.selectbox("Select Pair to Inspect", range(len(flat_data)), format_func=lambda i: f"{flat_data[i]['event']} ({flat_data[i]['consistency_score']})")
            
            if selected_row is not None:
                item = flat_data[selected_row]
                st.markdown(f"### {item['event']}")
                st.markdown(f"**Score:** {item['consistency_score']}")
                st.markdown(f"**Reasoning:** {item['reasoning']}")
                
                if item['contradictions']:
                    st.markdown("**Contradictions:**")
                    for c in item['contradictions']:
                        st.error(f"**{c['type'].upper()}** ({c['severity']}): {c['description']}")
                        st.caption(f"Ref: \"{c['quote_reference']}\"")
                else:
                    st.success("No contradictions found.")

    with tab2:
        st.subheader("Full Consistency Test (Reliability)")
        full_report = load_json("data/evaluated/full_consistency_report.json")
        
        if not full_report:
            st.warning("No full consistency report found in data/evaluated/full_consistency_report.json")
        else:
            df_full = pd.DataFrame(full_report)
            
            # Summary Metrics
            avg_mean = df_full['mean_score'].mean()
            avg_std = df_full['std_dev'].mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Overall Avg Score", f"{avg_mean:.2f}")
            col2.metric("Overall Avg Std Dev", f"{avg_std:.2f}")
            
            # Bar chart with error bars
            fig = px.bar(
                df_full, 
                x="event", 
                y="mean_score", 
                error_y="std_dev",
                color="source_b",
                barmode="group",
                title="Mean Consistency Score with Standard Deviation (5 Runs)",
                labels={"mean_score": "Mean Score", "event": "Event", "source_b": "Secondary Source"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_full)

# --- 5. Validation Metrics ---
elif page == "Validation Metrics":
    st.header("Validation Metrics")
    
    tab1, tab2 = st.tabs(["Kappa Test (Agreement)", "Prompt Comparison (Stability)"])
    
    with tab1:
        st.subheader("Inter-Rater Agreement (Cohen's Kappa)")
        kappa_data = load_json("data/evaluated/kappa_report.json")
        
        if kappa_data:
            col1, col2 = st.columns(2)
            col1.metric("Cohen's Kappa", f"{kappa_data['kappa']:.4f}")
            col2.metric("Raw Agreement", f"{kappa_data['raw_agreement']*100:.1f}%")
            
            st.markdown(f"**Interpretation:** {kappa_data['interpretation']}")
            
            # Confusion Matrix Heatmap
            cm = kappa_data['confusion_matrix']
            fig = px.imshow(cm, 
                            labels=dict(x="LLM Label", y="Human Label", color="Count"),
                            x=['Contradictory (0)', 'Consistent (1)'],
                            y=['Contradictory (0)', 'Consistent (1)'],
                            text_auto=True,
                            title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            st.json(kappa_data)
        else:
            st.warning("Kappa report not found.")
            
    with tab2:
        st.subheader("Prompt Strategy Comparison")
        prompt_data = load_json("data/evaluated/prompt_comparison_report.json")
        
        if prompt_data:
            summary = prompt_data['summary']
            winner = prompt_data['winner']
            
            st.success(f"🏆 **Winner:** {winner}")
            st.markdown(f"**Analysis:** The **{winner}** strategy demonstrated the highest stability (lowest Standard Deviation) across the test cases. Lower standard deviation indicates that the model produces more consistent scores when evaluating the same event multiple times.")
            
            # Bar Chart
            df_prompt = pd.DataFrame(list(summary.items()), columns=["Strategy", "Avg Std Dev"])
            fig = px.bar(df_prompt, x="Strategy", y="Avg Std Dev", color="Strategy", title="Stability Comparison (Lower is Better)", text_auto='.2f')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Stats per Strategy
            st.markdown("### Detailed Breakdown by Strategy")
            
            for strategy, details in prompt_data['details'].items():
                with st.expander(f"{strategy} Details"):
                    # Create a nice table for each strategy
                    strategy_data = []
                    for d in details:
                        strategy_data.append({
                            "Event": d['event'],
                            "Mean Score": f"{d['mean']:.1f}",
                            "Std Dev": f"{d['std_dev']:.2f}",
                            "Raw Scores": str(d['scores'])
                        })
                    st.dataframe(pd.DataFrame(strategy_data))
        else:
            st.warning("Prompt comparison report not found.")
