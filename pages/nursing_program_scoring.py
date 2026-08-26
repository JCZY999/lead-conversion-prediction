import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Nursing Program Scoring",page_icon="◎",layout="wide")

st.markdown("""<style>
.stApp{background:#07130f;color:#e7f5ee}
.hero{padding:2.3rem 0 1.2rem;border-top:3px solid #64f59e}
.eyebrow{color:#64f59e;font-weight:700;letter-spacing:.12em;font-size:.8rem}
.hero h1{font-size:3rem;margin:.5rem 0}
.hero p{color:#a8d5c4;font-size:1.125rem}
.note{background:#0f2f26;border-left:3px solid #64f59e;padding:1rem;border-radius:.5rem;margin:1.5rem 0}
.sidebar-meta{text-align:left;line-height:1.45}
.sidebar-meta p{margin:0 0 .85rem}
</style>""",unsafe_allow_html=True)

PROGRAMS=["Pre-Licensure BSN","Accelerated BSN","RN-to-BSN","MSN","RN-to-MSN","DNP","Nurse Practitioner","Other"]
EDUCATION=["High School/GED","Some College","Associate","Bachelor's","Master's","Doctorate"]
RN_LICENSE=["Yes","No","In Progress","Not Applicable"]
CREDENTIALS=["None","CNA","LPN/LVN","RN-ADN","RN-BSN","MSN/APRN","Other"]
GPA=["Below 2.5","2.5–2.99","3.0–3.49","3.5–3.79","3.8–4.0","Not Sure"]
PREREQUISITES=["All","Most","Some","None","Not Sure"]
START_TIME=["Next Available Term","Within 3 Months","3–6 Months","6–12 Months","12+ Months","Just Exploring"]
INTEREST=["Ready to Apply","Comparing Programs","Gathering Information","Exploring Nursing","Not Sure"]
EXPERIENCE=["None","<1 Year","1–2 Years","3–5 Years","6–10 Years","10+ Years"]
FORMAT=["Online","On-Campus","Hybrid","No Preference"]

def program_fit(program,education,rn_license,credential):
    education_rank={v:i for i,v in enumerate(EDUCATION)}
    if program=="Pre-Licensure BSN":
        return .95 if credential in ["None","CNA","LPN/LVN"] else .55
    if program=="Accelerated BSN":
        return .95 if education_rank[education]>=3 and credential not in ["RN-ADN","RN-BSN","MSN/APRN"] else .40
    if program=="RN-to-BSN":
        return .98 if rn_license=="Yes" and credential=="RN-ADN" else .30
    if program=="RN-to-MSN":
        return .95 if rn_license=="Yes" and credential in ["RN-ADN","RN-BSN"] else .25
    if program=="MSN":
        return .95 if rn_license=="Yes" and credential in ["RN-BSN","MSN/APRN"] else .25
    if program in ["DNP","Nurse Practitioner"]:
        return .95 if rn_license=="Yes" and credential=="MSN/APRN" else .20
    return .60

def nursing_lead_score(program,education,rn_license,credential,gpa,prerequisites,start_time,interest,experience,study_format):
    scores={
        "Program fit":program_fit(program,education,rn_license,credential),
        "Education":dict(zip(EDUCATION,[.45,.55,.70,.82,.92,1.0]))[education],
        "RN license":{"Yes":1.0,"In Progress":.72,"Not Applicable":.58,"No":.45}[rn_license],
        "Credential":dict(zip(CREDENTIALS,[.38,.50,.62,.78,.90,1.0,.55]))[credential],
        "GPA":dict(zip(GPA,[.25,.48,.68,.82,.95,.50]))[gpa],
        "Prerequisites":dict(zip(PREREQUISITES,[1.0,.82,.58,.25,.45]))[prerequisites],
        "Start readiness":dict(zip(START_TIME,[1.0,.94,.82,.65,.42,.25]))[start_time],
        "Interest":dict(zip(INTEREST,[1.0,.78,.58,.38,.30]))[interest],
        "Healthcare experience":dict(zip(EXPERIENCE,[.35,.48,.62,.78,.90,1.0]))[experience],
        "Study format":{"Online":.78,"On-Campus":.78,"Hybrid":.82,"No Preference":.90}[study_format],
    }
    weights={"Program fit":5,"Education":5,"RN license":5,"Credential":5,"GPA":4,"Prerequisites":4,"Start readiness":5,"Interest":5,"Healthcare experience":3,"Study format":3}
    weighted=sum(scores[k]*weights[k] for k in scores)/sum(weights.values())
    probability=1/(1+np.exp(-6*(weighted-.58)))
    return float(probability),scores

@st.cache_data
def generate_nursing_leads():
    rng=np.random.default_rng(42); n=800
    rows=[]
    probabilities=[]
    for i in range(n):
        answers={
            "Program":rng.choice(PROGRAMS,p=[.24,.13,.18,.12,.09,.06,.10,.08]),
            "Education":rng.choice(EDUCATION,p=[.18,.20,.22,.24,.12,.04]),
            "RN License":rng.choice(RN_LICENSE,p=[.44,.29,.17,.10]),
            "Credential":rng.choice(CREDENTIALS,p=[.23,.12,.11,.20,.19,.08,.07]),
            "GPA":rng.choice(GPA,p=[.08,.14,.29,.23,.16,.10]),
            "Prerequisites":rng.choice(PREREQUISITES,p=[.23,.31,.24,.10,.12]),
            "Start Time":rng.choice(START_TIME,p=[.20,.19,.23,.18,.10,.10]),
            "Interest":rng.choice(INTEREST,p=[.25,.30,.22,.15,.08]),
            "Healthcare Experience":rng.choice(EXPERIENCE,p=[.18,.16,.20,.20,.16,.10]),
            "Study Format":rng.choice(FORMAT,p=[.36,.22,.30,.12]),
        }
        probability,_=nursing_lead_score(*answers.values())
        rows.append({"Lead ID":f"NRS-{2000+i}",**answers})
        probabilities.append(probability)
    frame=pd.DataFrame(rows)
    frame["Predicted Score"]=probabilities
    frame["Enrolled"]=rng.binomial(1,frame["Predicted Score"])
    return frame

df=generate_nursing_leads()

nav_options=["Lead Intelligence","Score a Lead","Nursing Program Scoring","Model Evaluation","Methodology & Data"]
if "main_nav" not in st.session_state:
    st.session_state.main_nav="Nursing Program Scoring"
st.sidebar.markdown("## ◎ Lead Intelligence")
selected_page=st.sidebar.radio("Navigate",nav_options,key="main_nav")
st.sidebar.divider()
st.sidebar.markdown("""<div class="sidebar-meta">
<p><strong>Data Input:</strong><br>950 simulated data as example or upload your own data</p>
<p><strong>Model:</strong><br>Logistic Regression</p>
<p><strong>Training:</strong><br>Behavioral signals</p>
<p><strong>Created by:</strong><br>Jacob Chen</p>
</div>""",unsafe_allow_html=True)
if selected_page!="Nursing Program Scoring":
    st.switch_page("app.py")

st.markdown("""<div class='hero'><div class='eyebrow'>NURSING EDUCATION RECRUITMENT</div><h1>Nursing Program Scoring</h1><p>Score prospective students using program fit, academic readiness, credentials, timing, intent, and healthcare experience.</p></div>""",unsafe_allow_html=True)

tabs=st.tabs(["Score an Applicant","Applicant Analytics","Dataset Overview"])

with tabs[0]:
    st.markdown("### Nursing Program Interest and Readiness")
    left,right=st.columns(2)
    with left:
        program=st.selectbox("1. Which nursing program are you interested in?",PROGRAMS)
        education=st.selectbox("2. What is your highest level of education completed?",EDUCATION)
        rn_license=st.selectbox("3. Do you currently hold an active RN license?",RN_LICENSE,index=1)
        credential=st.selectbox("4. What nursing credential do you currently hold?",CREDENTIALS)
        gpa=st.selectbox("5. What is your approximate cumulative GPA?",GPA,index=2)
    with right:
        prerequisites=st.selectbox("6. Have you completed the required prerequisite courses?",PREREQUISITES,index=1)
        start_time=st.selectbox("7. When would you like to start your program?",START_TIME,index=1)
        interest=st.selectbox("8. What best describes your current level of interest?",INTEREST)
        experience=st.selectbox("9. How much healthcare experience do you have?",EXPERIENCE,index=1)
        study_format=st.selectbox("10. What is your preferred study format?",FORMAT,index=2)

    score,drivers=nursing_lead_score(program,education,rn_license,credential,gpa,prerequisites,start_time,interest,experience,study_format)
    tier="High Priority" if score>=.75 else "Qualified Lead" if score>=.50 else "Nurture"
    c1,c2,c3=st.columns(3)
    c1.metric("Enrollment Likelihood",f"{score:.1%}")
    c2.metric("Lead Tier",tier)
    c3.metric("Est. Program Value","$"+f"{score*45000:,.0f}")
    st.progress(score)
    recommendation={"High Priority":"Schedule an immediate advisor consultation and begin application support.","Qualified Lead":"Send program-match, prerequisite, and financial-aid guidance, then arrange advisor follow-up.","Nurture":"Share pathway information and maintain engagement until program fit and readiness improve."}[tier]
    st.markdown(f"<div class='note'><b>Recommendation:</b> {recommendation}</div>",unsafe_allow_html=True)
    driver_frame=pd.DataFrame({"Factor":drivers.keys(),"Readiness score":drivers.values()}).sort_values("Readiness score")
    fig=px.bar(driver_frame,x="Readiness score",y="Factor",orientation="h",template="plotly_dark",color_discrete_sequence=["#64f59e"],title="Applicant readiness drivers")
    fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f",xaxis_range=[0,1])
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.markdown("### Applicant Pool Analytics")
    view=df.copy()
    view["Tier"]=pd.cut(view["Predicted Score"],[-.01,.50,.75,1],labels=["Nurture","Qualified","High Priority"])
    predicted=(view["Predicted Score"]>=.5).astype(int)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Total Applicants",f"{len(view):,}")
    m2.metric("Actual Enrollment Rate",f"{view['Enrolled'].mean():.1%}")
    m3.metric("High Priority %",f"{(view['Tier']=='High Priority').mean():.1%}")
    m4.metric("Model Accuracy",f"{(predicted==view['Enrolled']).mean():.1%}")
    stats=view.groupby("Program",as_index=False).agg(Applicants=("Lead ID","count"),Avg_Score=("Predicted Score","mean"),Enrollment_Rate=("Enrolled","mean"),Pipeline_Value=("Predicted Score",lambda x:x.sum()*45000))
    fig=px.scatter(stats,x="Applicants",y="Avg_Score",size="Pipeline_Value",color="Program",template="plotly_dark",title="Program interest: lead quality vs volume")
    fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Top Applicants by Enrollment Likelihood")
    columns=["Lead ID","Program","Education","RN License","Credential","Start Time","Interest","Predicted Score","Enrolled"]
    top=view.nlargest(15,"Predicted Score")[columns].copy()
    top["Predicted Score"]=top["Predicted Score"].map(lambda value:f"{value:.1%}")
    top["Enrolled"]=top["Enrolled"].map(lambda value:"✓ Enrolled" if value else "Pending")
    st.dataframe(top,use_container_width=True,hide_index=True)

with tabs[2]:
    st.markdown("### Dataset Overview")
    st.markdown(f"**Total Applicants:** {len(df):,} | **Enrolled:** {df['Enrolled'].sum():,} | **Enrollment Rate:** {df['Enrolled'].mean():.1%}")
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(df,names="Program",template="plotly_dark",title="Program Interest")
        fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f")
        st.plotly_chart(fig,use_container_width=True)
        fig=px.bar(df.groupby("Education",as_index=False).size().sort_values("size"),x="size",y="Education",orientation="h",template="plotly_dark",color_discrete_sequence=["#64f59e"],title="Education Completed")
        fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f",showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.bar(df.groupby("Start Time",as_index=False).size().sort_values("size"),x="size",y="Start Time",orientation="h",template="plotly_dark",color_discrete_sequence=["#64f59e"],title="Intended Start Time")
        fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f",showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        fig=px.histogram(df,x="Predicted Score",nbins=30,template="plotly_dark",color_discrete_sequence=["#64f59e"],title="Score Distribution")
        fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f",showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
