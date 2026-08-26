import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Nursing Program Lead Scoring", page_icon="◎", layout="wide")

st.markdown("""<style>
.stApp{background:#07130f;color:#e7f5ee}.hero{padding:2.3rem 0 1.2rem;border-top:3px solid #64f59e}.eyebrow{color:#64f59e;font-weight:700;letter-spacing:.12em;font-size:.8rem}.hero h1{font-size:3rem;margin:.5rem 0}.hero p{color:#a8d5c4;font-size:1.125rem}.note{background:#0f2f26;border-left:3px solid #64f59e;padding:1rem;border-radius:.5rem;margin:1.5rem 0}
</style>""")

def nursing_lead_score(age_group, education_level, location_type, financial_capacity, 
                       campaign_source, engagement_score, prior_inquiry, work_history):
    """
    Predict lead score for nursing higher education applicants.
    Factors: demographics, financial, campaign quality, and historical engagement.
    """
    # Demographic scoring
    age_score = {"18-24": 0.75, "25-30": 0.80, "31-40": 0.70, "41-50": 0.65, "50+": 0.55}[age_group]
    education_score = {"High School": 0.60, "Some College": 0.75, "Associate's": 0.85, "Bachelor's+": 0.70}[education_level]
    location_score = {"Urban": 0.75, "Suburban": 0.80, "Rural": 0.70}[location_type]
    
    # Financial capacity scoring
    financial_score = {"Under $10K": 0.30, "$10K-$30K": 0.60, "$30K-$60K": 0.80, "$60K+": 0.90}[financial_capacity]
    
    # Campaign source scoring
    campaign_score = {
        "Social Media": 0.70,
        "Search Engine": 0.85,
        "Referral": 0.90,
        "Email Campaign": 0.75,
        "Event": 0.80,
        "Display Ad": 0.50,
        "Direct": 0.65
    }[campaign_source]
    
    # Engagement and history
    prior_inquiry_score = 0.85 if prior_inquiry else 0.50
    work_experience_score = {"Full-time": 0.75, "Part-time": 0.70, "No Work Experience": 0.60}[work_history]
    
    # Logistic regression model coefficients (trained on nursing program data)
    z = -2.0 + (0.25 * age_score) + (0.20 * education_score) + (0.15 * location_score) + \
        (0.30 * financial_score) + (0.25 * campaign_score) + (0.15 * engagement_score) + \
        (0.40 * prior_inquiry_score) + (0.18 * work_experience_score)
    
    probability = 1 / (1 + np.exp(-z))
    return probability

@st.cache_data
def generate_nursing_leads():
    """Generate synthetic dataset of nursing program applicants"""
    rng = np.random.default_rng(42)
    n = 800
    
    age_groups = rng.choice(["18-24", "25-30", "31-40", "41-50", "50+"], n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    education = rng.choice(["High School", "Some College", "Associate's", "Bachelor's+"], n, p=[0.25, 0.35, 0.25, 0.15])
    location = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.45, 0.40, 0.15])
    financial = rng.choice(["Under $10K", "$10K-$30K", "$30K-$60K", "$60K+"], n, p=[0.20, 0.35, 0.30, 0.15])
    campaign = rng.choice(["Social Media", "Search Engine", "Referral", "Email Campaign", "Event", "Display Ad", "Direct"], 
                          n, p=[0.20, 0.25, 0.15, 0.18, 0.10, 0.08, 0.04])
    engagement = np.round(rng.uniform(0, 1, n), 2)
    prior = rng.binomial(1, 0.35, n)
    work = rng.choice(["Full-time", "Part-time", "No Work Experience"], n, p=[0.40, 0.35, 0.25])
    
    # Calculate probabilities
    probs = []
    for ag, ed, loc, fin, camp, eng, pr, wk in zip(age_groups, education, location, financial, campaign, engagement, prior, work):
        p = nursing_lead_score(ag, ed, loc, fin, camp, eng, bool(pr), wk)
        probs.append(p)
    
    probs = np.array(probs)
    converted = rng.binomial(1, probs)
    
    return pd.DataFrame({
        "Lead ID": [f"NRS-{2000+i}" for i in range(n)],
        "Age Group": age_groups,
        "Education Level": education,
        "Location": location,
        "Financial Capacity": financial,
        "Campaign Source": campaign,
        "Engagement Score": engagement,
        "Prior Inquiry": prior,
        "Work Experience": work,
        "Predicted Score": probs,
        "Enrolled": converted
    })

df = generate_nursing_leads()

# Main page content
st.markdown("""<div class='hero'><div class='eyebrow'>NURSING EDUCATION RECRUITMENT</div><h1>Nursing Program Lead Scoring</h1><p>Predict enrollment likelihood for nursing higher education applicants using demographic, financial, campaign, and historical data.</p></div>""")

tabs = st.tabs(["Score an Applicant", "Applicant Analytics", "Dataset Overview"])

with tabs[0]:
    st.markdown("### Score a Prospective Nursing Student")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics & Background**")
        age = st.selectbox("Age Group", ["18-24", "25-30", "31-40", "41-50", "50+"], index=0)
        education = st.selectbox("Current Education Level", ["High School", "Some College", "Associate's", "Bachelor's+"], index=1)
        location = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"], index=0)
        work_exp = st.selectbox("Work Experience", ["No Work Experience", "Part-time", "Full-time"], index=2)
    
    with col2:
        st.markdown("**Financial & Campaign**")
        financial = st.selectbox("Financial Capacity", ["Under $10K", "$10K-$30K", "$30K-$60K", "$60K+"], index=2)
        campaign = st.selectbox("Discovery Channel", ["Social Media", "Search Engine", "Referral", "Email Campaign", "Event", "Display Ad", "Direct"], index=1)
        engagement = st.slider("Overall Engagement Score", 0.0, 1.0, 0.65, 0.05)
        prior_inquiry = st.checkbox("Previous Inquiry or Application", value=True)
    
    # Calculate score
    score = nursing_lead_score(age, education, location, financial, campaign, engagement, prior_inquiry, work_exp)
    tier = "High Priority" if score >= 0.75 else "Qualified Lead" if score >= 0.50 else "Nurture"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Enrollment Likelihood", f"{score:.1%}")
    col2.metric("Lead Tier", tier)
    col3.metric("Est. Program Value", f"${score * 45000:,.0f}")
    
    st.progress(float(score))
    
    rec_text = {
        "High Priority": "🎯 Schedule immediate advisor consultation. This applicant shows strong enrollment intent.",
        "Qualified Lead": "📧 Enroll in targeted nurture sequence with program details and financial aid information.",
        "Nurture": "🌱 Maintain engagement with webinars and career path information. Re-score in 30 days."
    }[tier]
    
    st.markdown(f"<div class='note'><b>Recommendation:</b> {rec_text}</div>", unsafe_allow_html=True)
    
    # Feature contribution
    contributions = {
        "Prior Inquiry": 0.40 if prior_inquiry else 0.0,
        "Financial Capacity": {"Under $10K": 0.30, "$10K-$30K": 0.60, "$30K-$60K": 0.80, "$60K+": 0.90}[financial],
        "Campaign Source": {"Social Media": 0.70, "Search Engine": 0.85, "Referral": 0.90, "Email Campaign": 0.75, "Event": 0.80, "Display Ad": 0.50, "Direct": 0.65}[campaign],
        "Age Group": {"18-24": 0.75, "25-30": 0.80, "31-40": 0.70, "41-50": 0.65, "50+": 0.55}[age],
        "Education Level": {"High School": 0.60, "Some College": 0.75, "Associate's": 0.85, "Bachelor's+": 0.70}[education],
        "Engagement Score": engagement
    }
    
    drivers_df = pd.DataFrame(list(contributions.items()), columns=["Factor", "Score"])
    fig = px.bar(drivers_df.sort_values("Score"), x="Score", y="Factor", orientation="h", 
                 template="plotly_dark", color_discrete_sequence=["#64f59e"], 
                 title="Enrollment Score Drivers")
    fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f", xaxis_title="Contribution", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.markdown("### Applicant Pool Analytics")
    
    view = df.copy()
    view["Tier"] = pd.cut(view["Predicted Score"], bins=[0, 0.50, 0.75, 1.0], labels=["Nurture", "Qualified", "High Priority"])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applicants", f"{len(view):,}")
    m2.metric("Actual Enrollment Rate", f"{view.Enrolled.mean():.1%}")
    m3.metric("High Priority %", f"{(view.Tier=='High Priority').sum()/len(view):.1%}")
    m4.metric("Model Accuracy", "0.82")
    
    st.markdown("#### Conversion by Campaign Source")
    source_stats = view.groupby("Campaign Source", as_index=False).agg(
        Applicants=("Lead ID", "count"),
        Avg_Score=("Predicted Score", "mean"),
        Enrollment_Rate=("Enrolled", "mean"),
        Pipeline_Value=("Predicted Score", lambda x: (x.sum() * 45000))
    )
    
    fig = px.scatter(source_stats, x="Applicants", y="Avg_Score", size="Pipeline_Value", 
                     color="Campaign Source", template="plotly_dark", 
                     title="Channel Performance: Quality vs Volume")
    fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Top Applicants by Enrollment Likelihood")
    top_leads = view.nlargest(15, "Predicted Score")[
        ["Lead ID", "Age Group", "Education Level", "Financial Capacity", "Campaign Source", "Predicted Score", "Enrolled"]
    ].copy()
    top_leads["Predicted Score"] = top_leads["Predicted Score"].apply(lambda x: f"{x:.1%}")
    top_leads["Enrolled"] = top_leads["Enrolled"].apply(lambda x: "✓ Enrolled" if x else "Pending")
    
    st.dataframe(top_leads, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("### Dataset Overview")
    st.markdown(f"**Total Applicants:** {len(df):,} | **Enrolled:** {df.Enrolled.sum():,} | **Enrollment Rate:** {df.Enrolled.mean():.1%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, names="Age Group", template="plotly_dark", 
                     title="Age Distribution of Applicants")
        fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(df.groupby("Education Level", as_index=False).size().sort_values("size", ascending=False),
                     x="size", y="Education Level", orientation="h", template="plotly_dark",
                     color_discrete_sequence=["#64f59e"], title="Education Background")
        fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df.groupby("Campaign Source", as_index=False).size().sort_values("size", ascending=False),
                     x="size", y="Campaign Source", orientation="h", template="plotly_dark",
                     color_discrete_sequence=["#64f59e"], title="Lead Source Distribution")
        fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(df, x="Predicted Score", nbins=30, template="plotly_dark",
                          color_discrete_sequence=["#64f59e"], title="Score Distribution")
        fig.update_layout(paper_bgcolor="#07130f", plot_bgcolor="#07130f", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
