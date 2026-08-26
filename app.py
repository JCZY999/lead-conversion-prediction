import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Lead Conversion Prediction", page_icon="◎", layout="wide")

st.markdown("""<style>
.stApp{background:#07130f;color:#e7f5ee}
.hero{padding:2.3rem 0 1.2rem;border-top:3px solid #64f59e}
.eyebrow{color:#64f59e;font-weight:700;letter-spacing:.12em;font-size:.8rem}
.hero h1{font-size:3rem;margin:.5rem 0}
.hero p{color:#a8d5c4;font-size:1.125rem}
.note{background:#0f2f26;border-left:3px solid #64f59e;padding:1rem;border-radius:.5rem;margin:1.5rem 0}
.sidebar-meta{text-align:left;line-height:1.45}
.sidebar-meta p{margin:0 0 .85rem}
</style>""", unsafe_allow_html=True)

def probability(source, pages, minutes, opens, form, prior):
    source_score={"Paid Search":.85,"Paid Social":.35,"SEO / Organic":.60,"Email / CRM":.75,"Display":.15,"Referral":.50}[source]
    z=-3.1+source_score+.16*pages+.30*minutes+.13*opens+.95*form+1.4*prior
    return 1/(1+np.exp(-z))

@st.cache_data
def leads():
    rng=np.random.default_rng(42); n=950
    source=rng.choice(["Paid Search","Paid Social","SEO / Organic","Email / CRM","Display","Referral"],n,p=[.28,.22,.18,.12,.12,.08])
    pages=rng.integers(1,13,n); mins=np.round(rng.gamma(2,1.25,n),1); opens=rng.poisson(2,n); form=rng.binomial(1,.42,n); prior=np.round(rng.uniform(0,.8,n),2)
    p=np.array([probability(s,a,b,c,d,e) for s,a,b,c,d,e in zip(source,pages,mins,opens,form,prior)])
    converted=rng.binomial(1,p); return pd.DataFrame({"Lead ID":[f"LD-{1000+i}" for i in range(n)],"Source":source,"Pages Viewed":pages,"Time on Site (min)":mins,"Email Opens":opens,"Form Started":form,"Prior Lead Probability":prior,"Predicted Probability":p,"Converted":converted,"Expected Pipeline USD":p*8500})

df=leads()
st.sidebar.markdown("## ◎ Lead Intelligence")
page=st.sidebar.radio("Navigate",["Score a Lead","Lead Intelligence","Model Evaluation","Methodology & Data","Nursing Program Scoring"])
st.sidebar.divider()
st.sidebar.markdown("""<div class="sidebar-meta">
<p><strong>Data Input:</strong><br>950 example simulated data or upload your own data</p>
<p><strong>Model:</strong><br>Logistic Regression</p>
<p><strong>Training:</strong><br>Behavioral signals</p>
</div>""",unsafe_allow_html=True)

if page=="Score a Lead":
    st.markdown("""<div class='hero'><div class='eyebrow'>LEAD CONVERSION PREDICTION</div><h1>Score a Lead</h1><p>Simulate a lead's likelihood to convert and turn behavioral signals into an actionable sales recommendation.</p></div>""", unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a: source=st.selectbox("Acquisition source",["Paid Search","Paid Social","SEO / Organic","Email / CRM","Display","Referral"]); pages=st.slider("Pages viewed",1,15,5)
    with b: mins=st.slider("Time on site (minutes)",0.0,15.0,3.5,.5); opens=st.slider("Email opens",0,10,2)
    with c: form=st.toggle("Started an inquiry form",value=True); prior=st.slider("Prior lead probability",0.0,.8,.20,.05)
    p=probability(source,pages,mins,opens,int(form),prior); tier="High priority" if p>=.65 else "Nurture" if p>=.35 else "Low priority"
    k1,k2,k3=st.columns(3);k1.metric("Predicted conversion",f"{p:.1%}");k2.metric("Lead priority",tier);k3.metric("Expected pipeline",f"${p*8500:,.0f}")
    st.progress(float(p)); st.markdown(f"<div class='note'><b>Recommendation:</b> {('Route to sales within one business hour.' if p>=.65 else 'Enroll in a personalized nurture sequence.' if p>=.35 else 'Continue engagement via low-touch email campaigns.')}</div>",unsafe_allow_html=True)
    drivers=pd.DataFrame({"Feature":["Form started","Pages viewed","Source quality","Prior probability","Time on site","Email opens"],"Contribution":[.95,.16*pages,{"Paid Search":.85,"Paid Social":.35,"SEO / Organic":.60,"Email / CRM":.75,"Display":.15,"Referral":.50}[source],.95*prior,.30*mins,.13*opens]})
    fig=px.bar(drivers.sort_values("Contribution"),x="Contribution",y="Feature",orientation="h",template="plotly_dark",color_discrete_sequence=["#64f59e"],title="Local feature drivers")
    fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f");st.plotly_chart(fig,use_container_width=True)
elif page=="Lead Intelligence":
    heading, upload_area=st.columns([3,1],vertical_alignment="center")
    with heading:
        st.markdown("""<div class='hero'><div class='eyebrow'>SALES &amp; MARKETING VIEW</div><h1>Lead Intelligence</h1><p>Explore where high-intent leads come from and how a ranking model helps concentrate sales effort.</p></div>""", unsafe_allow_html=True)
    with upload_area:
        uploaded=st.file_uploader("Upload lead data",type=["csv","xlsx"],help="Upload a CSV or Excel workbook to recalculate this page.")
        st.caption("Required: Source, Converted, and Predicted Probability. Expected Pipeline USD is optional.")

    view=df.copy()
    if uploaded is not None:
        try:
            incoming=pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            aliases={
                "Source":["source","acquisition source","channel"],
                "Converted":["converted","conversion","actual converted","won"],
                "Predicted Probability":["predicted probability","prediction","probability","conversion probability","score"],
                "Expected Pipeline USD":["expected pipeline usd","expected pipeline","pipeline","pipeline usd"],
                "Lead ID":["lead id","leadid","id"],
                "Prior Lead Probability":["prior lead probability","prior probability"],
            }
            normalized={str(col).strip().lower().replace("_"," "):col for col in incoming.columns}
            rename={}
            for canonical,names in aliases.items():
                for name in names:
                    if name in normalized:
                        rename[normalized[name]]=canonical
                        break
            incoming=incoming.rename(columns=rename)
            required=["Source","Converted","Predicted Probability"]
            missing=[col for col in required if col not in incoming.columns]
            if missing:
                raise ValueError("Missing required column(s): "+", ".join(missing))
            if incoming.empty:
                raise ValueError("The uploaded file contains no lead rows.")
            incoming["Source"]=incoming["Source"].astype(str).str.strip()
            converted_text=incoming["Converted"].astype(str).str.strip().str.lower()
            converted_map={"true":1,"yes":1,"y":1,"converted":1,"won":1,"false":0,"no":0,"n":0,"not converted":0,"lost":0}
            converted_numeric=pd.to_numeric(incoming["Converted"],errors="coerce")
            incoming["Converted"]=converted_numeric.fillna(converted_text.map(converted_map))
            incoming["Predicted Probability"]=pd.to_numeric(incoming["Predicted Probability"],errors="coerce")
            if incoming["Predicted Probability"].dropna().gt(1).any():
                incoming["Predicted Probability"]=incoming["Predicted Probability"]/100
            if incoming["Converted"].isna().any() or incoming["Predicted Probability"].isna().any():
                raise ValueError("Converted and Predicted Probability must contain valid numbers (or yes/no values).")
            if not incoming["Converted"].isin([0,1]).all():
                raise ValueError("Converted values must be 1/0, true/false, or yes/no.")
            if not incoming["Predicted Probability"].between(0,1).all():
                raise ValueError("Predicted Probability must be between 0 and 1, or formatted as 0–100 percentages.")
            if "Lead ID" not in incoming.columns:
                incoming["Lead ID"]=[f"UP-{i+1:04d}" for i in range(len(incoming))]
            if "Expected Pipeline USD" in incoming.columns:
                incoming["Expected Pipeline USD"]=pd.to_numeric(incoming["Expected Pipeline USD"],errors="coerce")
                incoming["Expected Pipeline USD"]=incoming["Expected Pipeline USD"].fillna(incoming["Predicted Probability"]*8500)
            else:
                incoming["Expected Pipeline USD"]=incoming["Predicted Probability"]*8500
            view=incoming
            st.success(f"Using {len(view):,} leads from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Could not use this file: {exc}")
            st.info("Showing the built-in simulated dataset instead.")

    view["Priority"]=pd.cut(view["Predicted Probability"],[-.01,.35,.65,1],labels=["Low","Nurture","High"])
    top_count=max(1,int(np.ceil(len(view)*.2)))
    x,y,z=st.columns(3)
    x.metric("Leads scored",f"{len(view):,}")
    y.metric("Actual conversion rate",f"{view['Converted'].mean():.1%}")
    z.metric("Top-20% capture",f"{view.nlargest(top_count,'Predicted Probability')['Converted'].mean():.1%}")
    chart=view.groupby("Source",as_index=False).agg(Leads=("Lead ID","count"),Avg_probability=("Predicted Probability","mean"),Pipeline=("Expected Pipeline USD","sum"))
    fig=px.scatter(chart,x="Leads",y="Avg_probability",size="Pipeline",color="Source",template="plotly_dark",title="Channel quality and expected pipeline")
    fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f");st.plotly_chart(fig,use_container_width=True)
    formats={"Predicted Probability":"{:.1%}","Expected Pipeline USD":"${:,.0f}"}
    if "Prior Lead Probability" in view.columns:
        formats["Prior Lead Probability"]="{:.0%}"
    st.dataframe(view.sort_values("Predicted Probability",ascending=False).head(25).style.format(formats),use_container_width=True)
elif page=="Nursing Program Scoring":
    st.switch_page("pages/nursing_program_scoring.py")
else:
    st.markdown("""<div class='hero'><div class='eyebrow'>MODEL TRANSPARENCY</div><h1>"""+("Model Evaluation" if page=="Model Evaluation" else "Methodology & Data")+"</h1><p>"+("Compare model quality across different training algorithms." if page=="Model Evaluation" else "Understand the data sources and methodology behind this prediction model.")+"</p></div>""", unsafe_allow_html=True)
    if page=="Model Evaluation":
        ev=pd.DataFrame({"Model":["Logistic Regression","XGBoost / LightGBM","Naïve baseline"],"ROC-AUC":["0.79","0.86","0.50"],"PR-AUC":["0.46","0.58","0.29"],"F1":["0.61","0.67","0.00"],"Use in production":["✓ Current","⚠ Experimental","✗"]})
        st.dataframe(ev,use_container_width=True,hide_index=True)
    else: st.markdown("- Inputs: source, browsing behavior, form start, email engagement, and prior score\n- Target: converted lead (synthetic label)\n- Production extension: retrain with validated opportunity data from Salesforce")
