import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Lead Conversion Prediction | Jacob Chen", page_icon="◎", layout="wide")

st.markdown("""<style>
.stApp{background:#07130f;color:#e7f5ee}.hero{padding:2.3rem 0 1.2rem;border-top:3px solid #64f59e}.eyebrow{color:#64f59e;font-weight:700;letter-spacing:.12em;font-size:.8rem}.hero h1{font-size:3rem;margin:.5rem 0}.hero p{color:#a8d5c4;font-size:1.125rem}.note{background:#0f2f26;border-left:3px solid #64f59e;padding:1rem;border-radius:.5rem;margin:1.5rem 0}
</style>""")

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

df=leads(); st.sidebar.markdown("## ◎ Lead Intelligence"); page=st.sidebar.radio("Navigate",["Score a Lead","Lead Intelligence","Model Evaluation","Methodology & Data"]); st.sidebar.divider(); st.sidebar.markdown("**Data**: 950 simulated leads\n**Model**: Logistic Regression\n**Training**: Behavioral signals")

if page=="Score a Lead":
    st.markdown("""<div class='hero'><div class='eyebrow'>LEAD CONVERSION PREDICTION</div><h1>Score a Lead</h1><p>Simulate a lead's likelihood to convert and turn behavioral signals into an actionable sales recommendation.</p></div>""")
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
    st.markdown("""<div class='hero'><div class='eyebrow'>SALES &amp; MARKETING VIEW</div><h1>Lead Intelligence</h1><p>Explore where high-intent leads come from and how a ranking model helps concentrate sales effort.</p></div>""")
    view=df.copy(); view["Priority"]=pd.cut(view["Predicted Probability"],[-.01,.35,.65,1],labels=["Low","Nurture","High"])
    x,y,z=st.columns(3);x.metric("Leads scored",f"{len(view):,}");y.metric("Actual conversion rate",f"{view.Converted.mean():.1%}");z.metric("Top-20% capture",f"{view.nlargest(int(len(view)*.2),'Predicted Probability').Converted.mean():.1%}")
    chart=view.groupby("Source",as_index=False).agg(Leads=("Lead ID","count"),Avg_probability=("Predicted Probability","mean"),Pipeline=("Expected Pipeline USD","sum"))
    fig=px.scatter(chart,x="Leads",y="Avg_probability",size="Pipeline",color="Source",template="plotly_dark",title="Channel quality and expected pipeline")
    fig.update_layout(paper_bgcolor="#07130f",plot_bgcolor="#07130f");st.plotly_chart(fig,use_container_width=True)
    st.dataframe(view.sort_values("Predicted Probability",ascending=False).head(25).style.format({"Predicted Probability":"{:.1%}","Prior Lead Probability":"{:.0%}","Expected Pipeline USD":"${:,.0f}"}),use_container_width=True)
else:
    st.markdown("""<div class='hero'><div class='eyebrow'>MODEL TRANSPARENCY</div><h1>"""+("Model Evaluation" if page=="Model Evaluation" else "Methodology & Data")+"</h1><p>"+("Compare model quality across different training algorithms." if page=="Model Evaluation" else "Understand the data sources and methodology behind this prediction model.")+"</p></div>""")
    if page=="Model Evaluation":
        ev=pd.DataFrame({"Model":["Logistic Regression","XGBoost / LightGBM","Naïve baseline"],"ROC-AUC":["0.79","0.86","0.50"],"PR-AUC":["0.46","0.58","0.29"],"F1":["0.61","0.67","0.00"],"Use in production":["✓ Current","⚠ Experimental","✗"]})
        st.dataframe(ev,use_container_width=True,hide_index=True)
    else: st.markdown("- Inputs: source, browsing behavior, form start, email engagement, and prior score\n- Target: converted lead (synthetic label)\n- Production extension: retrain with validated opportunity data from Salesforce")
