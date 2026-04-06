import streamlit as st, pandas as pd, numpy as np, sqlite3, hashlib, json
from datetime import datetime, date
import plotly.express as px, plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sleep Vs Stress", page_icon="🌙", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif}
.mcard{background:linear-gradient(145deg,#16213e,#0f3460);border:1px solid #1a4a7a;
       border-radius:14px;padding:18px 12px;text-align:center;margin:6px 0}
.mval{font-size:1.9rem;font-weight:800}
.mlbl{color:#78909c;font-size:.78rem;margin-top:4px;text-transform:uppercase}
.low{color:#4caf50!important}.medium{color:#ff9800!important}.high{color:#f44336!important}
.rec{background:#0d1b2a;border-left:4px solid #42a5f5;border-radius:8px;
     padding:13px 18px;margin:7px 0;color:#b0bec5}
.dis{background:#160e1e;border-left:4px solid #ce93d8;border-radius:8px;
     padding:10px 16px;margin:6px 0;color:#e0e0e0}
.alert{background:#2d0a0a;border:1px solid #f44336;border-radius:10px;
       padding:14px 20px;color:#ef9a9a;margin:10px 0;font-weight:600}
.talert{background:#0a1a2d;border:1px solid #42a5f5;border-radius:10px;
        padding:14px 20px;color:#90caf9;margin:10px 0}
.bye{background:linear-gradient(135deg,#0f3460,#16213e);border:1px solid #42a5f5;
     border-radius:16px;padding:28px 24px;text-align:center;color:#e3f2fd;margin:14px 0}
.sec{color:#90caf9;font-size:1.1rem;font-weight:700;border-bottom:1px solid #1e3a5f;
     padding-bottom:6px;margin:18px 0 12px}
.stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1);color:white;
                 border:none;border-radius:9px;padding:10px 24px;font-weight:700;width:100%}
h1,h2,h3{color:#e3f2fd}
</style>""", unsafe_allow_html=True)

# ── DATABASE ─────────────────────────────────────────────────────────────────
DB = "sleepiq.db"
def init_db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,password TEXT,age_group TEXT,age INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT,date TEXT,age INTEGER,
        age_group TEXT,sleep_hrs REAL,screen_time REAL,exercise INTEGER,caffeine INTEGER,
        stress_score REAL,stress_level TEXT,sleep_debt REAL,disorders TEXT,login_hour INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,date TEXT,rating INTEGER,
        helpful TEXT,improve TEXT,recommend TEXT,comment TEXT)""")
    try: c.execute("ALTER TABLE records ADD COLUMN login_hour INTEGER")
    except: pass
    c.commit(); c.close()

def save_feedback(fb):
    c = sqlite3.connect(DB)
    c.execute("""INSERT INTO feedback(username,date,rating,helpful,improve,recommend,comment)
        VALUES(?,?,?,?,?,?,?)""",
        (fb['u'],fb['dt'],fb['rating'],fb['helpful'],fb['improve'],fb['recommend'],fb['comment']))
    c.commit(); c.close()

def get_all_feedback():
    c = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM feedback ORDER BY date DESC", c)
    c.close(); return df

hp = lambda p: hashlib.sha256(p.encode()).hexdigest()

def reg(u,p,age,ag):
    c=sqlite3.connect(DB)
    try: c.execute("INSERT INTO users VALUES(?,?,?,?)",(u,hp(p),ag,age)); c.commit(); return True
    except: return False
    finally: c.close()

def login(u,p):
    c=sqlite3.connect(DB)
    r=c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,hp(p))).fetchone()
    c.close(); return r

def save(d):
    c=sqlite3.connect(DB)
    c.execute("""INSERT INTO records(username,date,age,age_group,sleep_hrs,screen_time,
        exercise,caffeine,stress_score,stress_level,sleep_debt,disorders,login_hour)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (d['u'],d['dt'],d['age'],d['ag'],d['sl'],d['sc'],d['ex'],d['ca'],
         d['score'],d['level'],d['debt'],d['dis'],d['hr']))
    c.commit(); c.close()

def history(u):
    c=sqlite3.connect(DB)
    df=pd.read_sql("SELECT * FROM records WHERE username=? ORDER BY date DESC",c,params=(u,))
    c.close(); return df

# ── ML MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def train():
    np.random.seed(42); n=3000
    sl=np.random.uniform(2,11,n); sc=np.random.uniform(0,12,n)
    ex=np.random.randint(0,2,n); ca=np.random.randint(0,2,n); ag=np.random.randint(0,4,n)
    score=(8-sl)*1.6+sc*0.55-ex*1.3+ca*0.9+ag*0.25+np.random.normal(0,.9,n)
    lbl=np.where(score<3,"Low",np.where(score<6,"Medium","High"))
    X=np.column_stack([sl,sc,ex,ca,ag]); s=int(n*.8)
    rf=RandomForestClassifier(150,max_depth=10,random_state=42)
    rf.fit(X[:s],lbl[:s])
    return rf, accuracy_score(lbl[s:],rf.predict(X[s:])), ["Sleep Hrs","Screen Time","Exercise","Caffeine","Age Group"]

RF,ACC,FEATS=train()
AGES={"Children (5–12 yrs)":0,"Teenagers (13–19 yrs)":1,"Adults (20–59 yrs)":2,"Older Adults (60+ yrs)":3}
REC_SL={"Children (5–12 yrs)":10,"Teenagers (13–19 yrs)":9,"Adults (20–59 yrs)":8,"Older Adults (60+ yrs)":7}

def predict(sl,sc,ex,ca,ag):
    X=np.array([[sl,sc,int(ex),int(ca),AGES.get(ag,1)]])
    lv=RF.predict(X)[0]; pr=RF.predict_proba(X)[0]
    score=sum(pr[i]*{"Low":2,"Medium":5,"High":8}[c] for i,c in enumerate(RF.classes_))
    return lv,round(score,1),dict(zip(RF.classes_,pr))

def debt(sl,ag): return round(max(0,REC_SL.get(ag,8)-sl)*365,1)

def impact(sl,lv,ex,ag):
    b=min(sl/REC_SL.get(ag,8),1.); p={"Low":0,"Medium":.15,"High":.35}[lv]; bo=.06 if ex else 0
    cl=lambda x:max(10,min(100,round(x*100)))
    return cl(b-p+bo),cl(b-p*1.2+bo),cl(b-p*.8+bo*1.5),cl(b-p*1.4+bo)

# ── RISK SCREENING DATA ───────────────────────────────────────────────────────
DIS={
 "Children (5–12 yrs)":[
  {"name":"Insomnia","base":30,"thr":8,
   "trigger":lambda sl,sc,ex,ca:sl<8,
   "ev":"Children sleeping <8 hrs have 2x higher insomnia risk (AAP,2023)",
   "sym":["Difficulty falling asleep","Waking frequently","Daytime sleepiness","Irritability"],
   "tx":["Fixed bedtime routine","No screens 1hr before bed","Relaxation activities","Doctor if severe"],
   "rec":["🛏️ Consistent bedtime even on weekends","📵 No screens 60 min before sleep","🎵 Calm music or bedtime story helps"]},
  {"name":"Night Terrors","base":20,"thr":9,
   "trigger":lambda sl,sc,ex,ca:sl<9 or sc>4,
   "ev":"Sleep deprivation & high screen exposure increase night terrors (Sleep Med,2022)",
   "sym":["Sudden screaming during sleep","Fear without remembering","Fast breathing & sweating"],
   "tx":["Regular sleep schedule","Reduce daytime stress","Safe sleeping environment","Paediatrician if frequent"],
   "rec":["🌙 Keep afternoons calm — overstimulation worsens terrors","🧸 Consistent pre-sleep ritual","📅 Log occurrences to find triggers"]}
 ],
 "Teenagers (13–19 yrs)":[
  {"name":"Anxiety Disorder","base":35,"thr":8,
   "trigger":lambda sl,sc,ex,ca:sl<8 or sc>5,
   "ev":"Teens <8 hrs sleep have 3x anxiety risk; screen>5hrs doubles it (NIMH,2023)",
   "sym":["Constant worry","Restlessness","Trouble sleeping","Fast heartbeat"],
   "tx":["CBT techniques","Deep breathing exercises","30 min exercise daily","Counselling if persistent"],
   "rec":["🫁 4-7-8 breathing before bed","📓 Journal 3 worries — externalising reduces rumination","📱 No social media after 9 PM"]},
  {"name":"Depression","base":25,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<7 and not ex,
   "ev":"Teens with sleep deprivation + no exercise have elevated depression risk (WHO,2023)",
   "sym":["Feeling sad most of the time","Loss of interest in activities","Sleep problems","Low energy"],
   "tx":["Consistent sleep routine","Social support","Professional counselling","Healthy diet & exercise"],
   "rec":["☀️ Sunlight within 30 mins of waking boosts serotonin","🤝 Talk to one trusted person today","🏃 Exercise is as effective as mild antidepressants"]},
  {"name":"Delayed Sleep Phase Syndrome","base":28,"thr":8,
   "trigger":lambda sl,sc,ex,ca:sc>5 or sl<7,
   "ev":"High evening screen exposure disrupts circadian rhythm in teens (J Sleep Research,2022)",
   "sym":["Sleeping very late","Difficulty waking early","Daytime fatigue","Normal sleep once asleep"],
   "tx":["Shift bedtime 15 min earlier per week","Limit phone after 9 PM","Morning sunlight exposure"],
   "rec":["📱 Phone in another room at 10 PM","🌅 Bright window within 15 mins of waking","🚫 No naps after 3 PM"]}
 ],
 "Adults (20–59 yrs)":[
  {"name":"Sleep Apnea","base":30,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<7,
   "ev":"Adults <7 hrs sleep with high stress have higher sleep apnea risk (AHA,2023)",
   "sym":["Loud snoring","Breathing pauses during sleep","Morning headaches","Extreme daytime tiredness"],
   "tx":["Weight management","No alcohol 3hrs before bed","CPAP machine if prescribed","Side sleeping"],
   "rec":["🍷 No alcohol 3hrs before bed — relaxes throat muscles","🛌 Sleep on your side","🩺 Loud snoring + daytime fatigue = request a sleep study"]},
  {"name":"Hypertension","base":32,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<7 or not ex,
   "ev":"Poor sleep + no exercise doubles hypertension risk (AHA,2023)",
   "sym":["Frequent headaches","Dizziness","Fatigue","Often no visible symptoms"],
   "tx":["Stress reduction techniques","150 min exercise/week","Low sodium DASH diet","Medication if required"],
   "rec":["🧘 10 min mindfulness at lunch reduces BP","🥗 Under 2300mg sodium/day","💊 See doctor if persistent headaches"]},
  {"name":"Chronic Insomnia","base":28,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<6,
   "ev":"Adults <6 hrs for 2+ weeks meet chronic insomnia criteria (DSM-5)",
   "sym":["Difficulty sleeping for weeks","Poor concentration","Mood changes","Daytime impairment"],
   "tx":["Sleep hygiene improvements","No caffeine after 2 PM","CBT-I therapy","Doctor if >3 months"],
   "rec":["☕ No caffeine after 2 PM — 6-8hr half-life","🌡️ Bedroom 18–20°C triggers sleep","📋 CBT-I outperforms sleeping pills long-term"]}
 ],
 "Older Adults (60+ yrs)":[
  {"name":"Cognitive Decline Risk","base":30,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<6 and not ex,
   "ev":"Chronic sleep deprivation increases cognitive decline & dementia risk (Lancet,2022)",
   "sym":["Increased forgetfulness","Confusion or disorientation","Fragmented sleep","Difficulty with familiar tasks"],
   "tx":["Structured daily routine","Cognitive exercises (puzzles,reading)","Physical activity","Medical evaluation if persistent"],
   "rec":["🧩 Daily puzzles protect cognitive reserve","🚶 30 min walk reduces dementia risk 35% (Lancet)","📅 Fixed routine supports memory consolidation"]},
  {"name":"Restless Legs Syndrome","base":25,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<7 or ca,
   "ev":"RLS increases with age; caffeine & poor sleep are known aggravators (AASM,2022)",
   "sym":["Urge to move legs at night","Leg discomfort/crawling sensation","Worse at rest","Sleep disturbance"],
   "tx":["Regular moderate exercise","Iron supplements if deficient","No caffeine/alcohol at night","Leg stretches before bed"],
   "rec":["🦵 10 min leg stretches before bed","☕ No caffeine after noon for older adults","🩺 Check ferritin (iron) levels — low iron triggers RLS"]},
  {"name":"Chronic Insomnia","base":28,"thr":7,
   "trigger":lambda sl,sc,ex,ca:sl<6,
   "ev":"Insomnia affects 30–48% of older adults; worsens with poor sleep hygiene (Sleep Med Rev,2023)",
   "sym":["Difficulty falling/staying asleep","Waking too early","Poor concentration & mood changes","Daytime fatigue"],
   "tx":["Sleep hygiene improvements","Limit caffeine & naps","CBT-I preferred over sleeping pills","Medical advice if persistent"],
   "rec":["🌅 Morning sunlight resets circadian rhythm","💊 Avoid sleeping pills — fall risk is higher in seniors","🧘 Gentle yoga improves senior sleep quality 37%"]}
 ]
}

def risk_score(sl,sc,ex,ca,ag,base,thr):
    deficit=max(0,REC_SL.get(ag,8)-sl)
    s=base+min(deficit*10,30)+min(sc*2,15)-(8 if ex else 0)+(5 if ca else 0)
    return min(100,max(5,round(s)))

def risk_label(s): return "High Risk" if s>=65 else "Moderate Risk" if s>=40 else "Low Risk"

def get_risks(sl,sc,ex,ca,ag):
    out=[]
    for d in DIS.get(ag,[]):
        if d["trigger"](sl,sc,ex,ca):
            rs=risk_score(sl,sc,ex,ca,ag,d["base"],d["thr"])
            out.append({**d,"rs":rs,"rl":risk_label(rs)})
    return sorted(out,key=lambda x:x["rs"],reverse=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def recs(sl,lv,sc,ex,ca,ag):
    r=[]
    if sl<6 and lv=="High": r+=["🛏️ Critical: Under 6 hrs + high stress. Target 8 hrs tonight.","📵 No screens after 9 PM — blue light blocks melatonin.","🕯️ Wind-down: dim lights → warm shower → book → sleep."]
    elif sl>=7 and lv=="High": r+=["🧠 Enough sleep but stress still high — check for anxiety.","🫁 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s.","📓 Write 3 worries before bed — externalising reduces cortisol."]
    elif lv=="Medium": r+=["⚠️ Moderate stress. Small changes now prevent High stress tomorrow.","🏃 20 min walk today measurably improves tonight's sleep."]
    else: r+=["✅ Healthy stress level! Keep your routine.","💪 Maintain sleep schedule even on weekends."]
    if ca: r.append("☕ Caffeine lasts 6–8 hrs. Avoid after 2 PM.")
    if sc>5: r.append("📱 Over 5 hrs screen time worsens sleep. Set a 2-hr limit.")
    if not ex: r.append("🏃 No exercise — even 15 min walk cuts cortisol by 20%.")
    ag_r={"Teenagers (13–19 yrs)":"📚 Teens need 8–10 hrs. Sleep consolidates memory.",
          "Children (5–12 yrs)":"🌙 Children need 9–12 hrs. Consistent bedtime is key.",
          "Adults (20–59 yrs)":"💼 Schedule 10 min mindfulness during your workday.",
          "Older Adults (60+ yrs)":"🌅 Morning sunlight within 30 min of waking regulates body clock."}
    if ag in ag_r: r.append(ag_r[ag])
    return r



# ── TIME ALERTS ───────────────────────────────────────────────────────────────
def time_alert(h, ag):
    if h >= 22 or h < 2: return "🌙", f"Late night! You need {REC_SL.get(ag,8)} hrs sleep. Finish quickly and rest now!"
    if h < 6:  return "⚠️", f"It's {h:02d}:00 — extremely late! Staying up now seriously increases tomorrow's stress score."
    if h < 12: return "☀️", "Good morning! Great time to check in. Morning tracking helps you plan a low-stress day."
    if h < 17: return "🌤️", "Afternoon check-in. Don't skip lunch — hunger spikes cortisol and worsens stress."
    return "🌆", f"Evening check-in. Start winding down in {23-h} hour(s) for optimal sleep tonight."

def goodbye_msg(lv,sl,ag,hr):
    rec=REC_SL.get(ag,8)
    if lv=="High":
        h="⚠️ High Stress — Your Tonight Plan"
        b=f"You logged {sl} hrs vs recommended {rec} hrs. Be in bed by {max(21,23-rec):02d}:00. Avoid screens & caffeine now."
    elif lv=="Medium":
        h="🟠 Moderate Stress — Small Steps Tonight"
        b=f"Sleep {rec} hrs consistently for 3 nights to normalise cortisol. Aim for bed by {max(22,24-rec):02d}:00."
    else:
        h="✅ You're Doing Well!"; b=f"Low stress, {sl} hrs sleep. Keep it consistent even on weekends."
    if hr>=22 or hr<6: b+=f"\n\n🌙 Night Alert: You checked in at {hr:02d}:xx. Please sleep now."
    return h,b

# ── TIMELINE ──────────────────────────────────────────────────────────────────
def timeline(df):
    if df.empty: return
    rows=df[['date','sleep_hrs']].head(60).sort_values('date')
    cells="".join(f'<span style="display:inline-block;width:14px;height:28px;border-radius:3px;margin:1px;background:{"#4caf50" if r.sleep_hrs>=7 else "#ff9800" if r.sleep_hrs>=5 else "#f44336"}" title="{r.date}:{r.sleep_hrs}hrs"></span>' for _,r in rows.iterrows())
    st.markdown(f'<div style="margin:10px 0"><p style="color:#78909c;font-size:.8rem">SLEEP TIMELINE &nbsp;<span style="color:#4caf50">■</span>Good(7+) &nbsp;<span style="color:#ff9800">■</span>Moderate(5-7) &nbsp;<span style="color:#f44336">■</span>Poor(&lt;5)</p>{cells}</div>',unsafe_allow_html=True)

# ── PAGES ─────────────────────────────────────────────────────────────────────
def page_login():
    st.markdown("# 🌙 Sleep Vs Stress"); st.markdown("*Know your sleep. Control your stress.*"); st.markdown("---")
    h=datetime.now().hour; em,msg=time_alert(h,"Adults (20–59 yrs)")
    st.markdown(f'<div class="talert">{em} {msg}</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("### 🔑 Login")
        u=st.text_input("Username",key="lu"); p=st.text_input("Password",type="password",key="lp")
        if st.button("Login →"):
            row=login(u,p)
            if row: st.session_state.user={"u":row[0],"ag":row[2],"age":row[3],"hr":h,"lv":None,"sl":None}; st.rerun()
            else: st.error("Invalid credentials")
    with c2:
        st.markdown("### 📝 Register")
        nu=st.text_input("Username",key="ru"); np_=st.text_input("Password",type="password",key="rp")
        age=st.number_input("Age",5,100,19,key="ra")
        ag=("Children (5–12 yrs)" if age<=12 else "Teenagers (13–19 yrs)" if age<=19 else "Adults (20–59 yrs)" if age<=59 else "Older Adults (60+ yrs)")
        st.info(f"Age group: **{ag}**")
        if st.button("Register →"):
            st.success("✅ Done! Please login.") if reg(nu,np_,age,ag) else st.error("Username taken.")

def page_analyze(user):
    h=user['hr']; em,msg=time_alert(h,user['ag'])
    st.markdown(f'<div class="talert">{em} {msg}</div>',unsafe_allow_html=True)
    st.markdown(f"## 📊 Analysis &nbsp;|&nbsp; {user['ag']} &nbsp;|&nbsp; Age {user['age']}")
    c1,c2=st.columns(2)
    with c1:
        sl=st.slider("😴 Sleep last night (hrs)",0.0,12.0,6.0,.5)
        sc=st.slider("📱 Screen time (hrs)",0.0,16.0,4.0,.5)
    with c2:
        ex=st.checkbox("🏃 Exercised today?"); ca=st.checkbox("☕ Caffeine today?")

    if st.button("🔍 Analyze My Stress"):
        lv,score,proba=predict(sl,sc,ex,ca,user['ag'])
        dbt=debt(sl,user['ag']); fo,em_,ph,re=impact(sl,lv,ex,user['ag'])
        user['lv']=lv; user['sl']=sl; st.session_state.user=user

        # Metrics
        st.markdown("---"); css="low" if lv=="Low" else "medium" if lv=="Medium" else "high"
        cols=st.columns(4)
        for col,lbl,val in zip(cols,["Stress Score","Stress Level","Sleep Debt","Sleep Hrs"],
                               [f"{score}/10",lv,f"{round(dbt/24,1)}d",f"{sl}hrs"]):
            col.markdown(f'<div class="mcard"><div class="mval {css}">{val}</div><div class="mlbl">{lbl}</div></div>',unsafe_allow_html=True)
        if lv=="High":
            st.markdown(f'<div class="alert">⚠️ HIGH STRESS — {round(dbt/24,1)} days sleep debt. Act tonight.</div>',unsafe_allow_html=True)

        # Stress pie chart
        st.markdown('<p class="sec">🥧 Stress Probability</p>',unsafe_allow_html=True)
        fig=px.pie(values=list(proba.values()),names=list(proba.keys()),hole=.45,
                   color=list(proba.keys()),color_discrete_map={"Low":"#4caf50","Medium":"#ff9800","High":"#f44336"})
        fig.update_traces(textinfo='percent+label',textfont_size=13)
        fig.update_layout(paper_bgcolor="#0d1b2a",font_color="white",showlegend=False,
                          annotations=[dict(text=f"<b>{lv}</b>",font_size=18,font_color="white",showarrow=False)])
        st.plotly_chart(fig,use_container_width=True)

        # Brain/Body donuts
        st.markdown('<p class="sec">🧠 Brain & Body Impact</p>',unsafe_allow_html=True)
        bc=st.columns(4)
        for col,lbl,val in zip(bc,["Focus","Emotional","Physical","Resilience"],[fo,em_,ph,re]):
            clr="#4caf50" if val>=70 else "#ff9800" if val>=50 else "#f44336"
            fig2=go.Figure(go.Pie(values=[val,100-val],hole=.6,marker_colors=[clr,"#1a2332"],textinfo="none",hoverinfo="skip"))
            fig2.update_layout(paper_bgcolor="#0d1b2a",showlegend=False,height=150,margin=dict(t=5,b=5,l=5,r=5),
                               annotations=[dict(text=f"<b>{val}%</b>",font_size=15,font_color=clr,showarrow=False)])
            col.markdown(f"<p style='color:#78909c;font-size:.75rem;text-align:center'>{lbl}</p>",unsafe_allow_html=True)
            col.plotly_chart(fig2,use_container_width=True)

        # Gauge
        fig3=go.Figure(go.Indicator(mode="gauge+number",value=score,
            title={"text":"Stress Score (RF Model)","font":{"color":"white","size":13}},
            number={"font":{"color":"white","size":34}},
            gauge={"axis":{"range":[0,10],"tickcolor":"white"},"bar":{"color":"#42a5f5","thickness":.25},
                   "steps":[{"range":[0,3],"color":"#1b5e20"},{"range":[3,6],"color":"#e65100"},{"range":[6,10],"color":"#b71c1c"}],
                   "threshold":{"line":{"color":"white","width":3},"thickness":.75,"value":score}}))
        fig3.update_layout(paper_bgcolor="#0d1b2a",font_color="white",height=250)
        st.plotly_chart(fig3,use_container_width=True)

        # Recommendations
        st.markdown('<p class="sec">💡 Personalised Recommendations</p>',unsafe_allow_html=True)
        for r in recs(sl,lv,sc,ex,ca,user['ag']):
            st.markdown(f'<div class="rec">{r}</div>',unsafe_allow_html=True)

        # Lifestyle Risk Screening (High only)
        if lv=="High":
            risks=get_risks(sl,sc,ex,ca,user['ag'])
            st.markdown('<p class="sec">🔬 Lifestyle Risk Screening</p>',unsafe_allow_html=True)
            st.markdown("""<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;
                padding:12px 18px;margin:8px 0;color:#78909c;font-size:.82rem">
                ⚕️ <strong style="color:#90caf9">DISCLAIMER:</strong>
                This is <strong>lifestyle-based risk screening</strong>, NOT medical diagnosis.
                Risk scores reflect lifestyle associations from medical literature.
                <strong>Consult a doctor for any health concerns.</strong></div>""",unsafe_allow_html=True)
            if risks:
                # Risk bar chart
                fig4=go.Figure(go.Bar(x=[d['rs'] for d in risks],y=[d['name'] for d in risks],orientation='h',
                    marker_color=["#f44336" if d['rs']>=65 else "#ff9800" if d['rs']>=40 else "#4caf50" for d in risks],
                    text=[f"{d['rs']}% — {d['rl']}" for d in risks],textposition='outside',textfont=dict(color='white',size=11)))
                fig4.update_layout(paper_bgcolor="#0d1b2a",plot_bgcolor="#0d1b2a",font_color="white",
                    title=dict(text="Lifestyle Risk Association Score (0–100)",font=dict(color="#90caf9",size=12)),
                    xaxis=dict(range=[0,115],showgrid=False),yaxis=dict(autorange="reversed"),
                    height=60+len(risks)*55,margin=dict(l=10,r=80,t=35,b=10))
                st.plotly_chart(fig4,use_container_width=True)

                # Individual cards
                for d in risks:
                    bc_="#f44336" if d['rs']>=65 else "#ff9800" if d['rs']>=40 else "#4caf50"
                    icon="🔴" if d['rs']>=65 else "🟠" if d['rs']>=40 else "🟢"
                    with st.expander(f"{icon} {d['name']} — {d['rs']}/100 ({d['rl']})"):
                        st.markdown(f'<div style="background:#0a1f0a;border-left:3px solid #4caf50;border-radius:6px;padding:8px 14px;margin-bottom:10px;color:#a5d6a7;font-size:.82rem">📖 {d["ev"]}</div>',unsafe_allow_html=True)
                        st.markdown(f'<div style="background:#0d1b2a;border-radius:8px;padding:10px 14px;margin-bottom:10px"><div style="background:#1a2332;border-radius:6px;height:10px"><div style="background:{bc_};border-radius:6px;height:10px;width:{d["rs"]}%"></div></div><div style="color:{bc_};font-weight:700;margin-top:5px">{d["rs"]}/100 — {d["rl"]}</div></div>',unsafe_allow_html=True)
                        cc1,cc2=st.columns(2)
                        with cc1:
                            st.markdown("**⚠️ Symptoms:**")
                            for s in d['sym']: st.markdown(f"- {s}")
                        with cc2:
                            st.markdown("**💊 Treatment:**")
                            for t in d['tx']: st.markdown(f"- {t}")
                        st.markdown("**🎯 Recommendations:**")
                        for r in d.get('rec',[]): st.markdown(f'<div class="dis">💡 {r}</div>',unsafe_allow_html=True)

                if sum(1 for d in risks if d['rs']>=65)>0:
                    st.markdown('<div class="alert">🩺 High risk detected. Please discuss your sleep patterns with a doctor.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rec" style="border-left-color:#4caf50">✅ Risk screening activates only at High stress. Your level: <strong>{lv}</strong> — you\'re in a safe zone.</div>',unsafe_allow_html=True)

        # Save
        save({"u":user['u'],"dt":str(date.today()),"age":user['age'],"ag":user['ag'],
              "sl":sl,"sc":sc,"ex":int(ex),"ca":int(ca),"score":score,"level":lv,
              "debt":dbt,"dis":json.dumps([d['name'] for d in get_risks(sl,sc,ex,ca,user['ag'])] if lv=="High" else []),
              "hr":user['hr']})
        st.success("✅ Record saved!")

        

def chart_analysis(title, emoji, points):
     st.markdown(f"**{emoji} {title}**")
     for p in points:st.markdown(f"""
<div style="background:#0d1b2a;border-left:3px solid {p['color']};
     border-radius:8px;padding:10px 16px;margin:6px 0">
<span style="font-size:1rem">{p['icon']}</span>
<strong style="color:{p['color']}">&nbsp;{p['label']}:</strong>
<span style="color:#b0bec5;font-size:.88rem"> {p['text']}</span>
</div>""", unsafe_allow_html=True)
    
def page_history(user):
    st.markdown("## 📈 Your Sleep & Stress History")
    df = history(user['u'])
    if df.empty:
        st.info("No records yet. Run your first analysis!"); return

    # Fix date format
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

    # ── Sleep Timeline ───────────────────────────────────────────────────────
    timeline(df)
    good  = len(df[df['sleep_hrs'] >= 7])
    mod   = len(df[(df['sleep_hrs'] >= 5) & (df['sleep_hrs'] < 7)])
    poor  = len(df[df['sleep_hrs'] < 5])
    total = len(df)
    chart_analysis("What Your Sleep Timeline Means", "🗓️", [
        {"icon":"🟢","color":"#4caf50","label":"Green days",
         "text":f"You had {good} day(s) with 7+ hours of sleep — these are your healthy days. Your body recovered well on these nights."},
        {"icon":"🟠","color":"#ff9800","label":"Orange days",
         "text":f"You had {mod} day(s) with 5–7 hours — moderate sleep. You may have felt slightly tired or unfocused the next day."},
        {"icon":"🔴","color":"#f44336","label":"Red days",
         "text":f"You had {poor} day(s) with under 5 hours — poor sleep. These are the days most likely to cause high stress and health risks."},
        {"icon":"💡","color":"#90caf9","label":"Goal",
         "text":f"Aim to turn as many days green as possible. Out of your {total} recorded days, {round(good/total*100)}% were healthy sleep days."},
    ])

    # ── Pie Charts ───────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names="stress_level", color="stress_level",
                     title="Stress Level Distribution",
                     color_discrete_map={"Low":"#4caf50","Medium":"#ff9800","High":"#f44336"})
        fig.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        # Stress pie analysis
        high_pct  = round(len(df[df['stress_level']=="High"]) / total * 100)
        med_pct   = round(len(df[df['stress_level']=="Medium"]) / total * 100)
        low_pct   = round(len(df[df['stress_level']=="Low"]) / total * 100)
        dom = "High" if high_pct>=med_pct and high_pct>=low_pct else "Medium" if med_pct>=low_pct else "Low"
        dom_msg = ("⚠️ You are frequently under high stress. Prioritise sleep and reduce screen time immediately."
                   if dom=="High" else
                   "You are mostly in moderate stress. Small improvements to your sleep routine will help."
                   if dom=="Medium" else
                   "Great news — you are mostly in a healthy, low-stress state. Keep it up!")
        chart_analysis("What Your Stress Pie Chart Means", "🥧", [
            {"icon":"🔴","color":"#f44336","label":f"High stress",
             "text":f"{high_pct}% of your days — these are days your body was under serious strain."},
            {"icon":"🟠","color":"#ff9800","label":f"Medium stress",
             "text":f"{med_pct}% of your days — manageable but needs attention."},
            {"icon":"🟢","color":"#4caf50","label":f"Low stress",
             "text":f"{low_pct}% of your days — healthy days where your body recovered well."},
            {"icon":"🎯","color":"#90caf9","label":"Your situation",
             "text": dom_msg},
        ])

    with c2:
        bins = pd.cut(df['sleep_hrs'], bins=[0,5,7,9,12], labels=["<5hrs","5-7hrs","7-9hrs","9+hrs"])
        bc = bins.value_counts().reset_index(); bc.columns = ["Range","Count"]
        fig2 = px.pie(bc, names="Range", values="Count", title="Sleep Duration Distribution",
                      color_discrete_sequence=["#f44336","#ff9800","#4caf50","#42a5f5"])
        fig2.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

        # Sleep pie analysis
        avg_sl = round(df['sleep_hrs'].mean(), 1)
        rec_sl = REC_SL.get(user['ag'], 8)
        sl_msg = ("You are sleeping well on most days — keep this going!" if avg_sl >= rec_sl
                  else f"Your average sleep is {avg_sl} hrs but your body needs {rec_sl} hrs. You are losing {round((rec_sl-avg_sl)*365/24,1)} days of sleep per year!")
        chart_analysis("What Your Sleep Duration Pie Means", "💤", [
            {"icon":"🔴","color":"#f44336","label":"Under 5 hrs",
             "text":"Dangerously low sleep. High risk of stress, memory problems, and health issues."},
            {"icon":"🟠","color":"#ff9800","label":"5–7 hrs",
             "text":"Below recommended. You may feel okay but your body is slowly accumulating sleep debt."},
            {"icon":"🟢","color":"#4caf50","label":"7–9 hrs",
             "text":"Healthy range for most adults. Your brain and body recover properly at this level."},
            {"icon":"📊","color":"#90caf9","label":"Your average",
             "text":f"You average {avg_sl} hrs/night. {sl_msg}"},
        ])

    # ── Trend Line Chart ─────────────────────────────────────────────────────
    if df['date'].nunique() < 2:
        st.info("📅 Record data on at least 2 different days to see your Sleep vs Stress Trend chart.")
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['date'], y=df['sleep_hrs'], name="Sleep Hrs",
                                  mode="lines+markers", line=dict(color="#42a5f5", width=2)))
        fig3.add_trace(go.Scatter(x=df['date'], y=df['stress_score'], name="Stress Score",
                                  mode="lines+markers", line=dict(color="#f44336", width=2), yaxis="y2"))
        fig3.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a", font_color="white",
            title="Sleep vs Stress Trend — How Your Sleep Affects Your Stress Over Time",
            xaxis=dict(title="Date", color="white", type="category"),
            yaxis=dict(title="Sleep Hrs", color="#42a5f5"),
            yaxis2=dict(title="Stress Score", overlaying="y", side="right", color="#f44336"),
            legend=dict(font=dict(color="white")))
        st.plotly_chart(fig3, use_container_width=True)

        # Trend analysis
        avg_stress = round(df['stress_score'].mean(), 1)
        best_day   = df.loc[df['sleep_hrs'].idxmax(), 'date']
        worst_day  = df.loc[df['sleep_hrs'].idxmin(), 'date']
        corr       = df[['sleep_hrs','stress_score']].corr().iloc[0,1]
        corr_msg   = ("When your sleep goes up, your stress goes down — the chart clearly shows this pattern. This is exactly what medical research predicts."
                      if corr < -0.3 else
                      "Your sleep and stress don't show a strong pattern yet — add more daily records to see a clearer trend.")
        chart_analysis("What Your Sleep vs Stress Trend Means", "📈", [
            {"icon":"🔵","color":"#42a5f5","label":"Blue line (Sleep Hrs)",
             "text":"Shows how many hours you slept each day. When this line goes UP, you slept more that night."},
            {"icon":"🔴","color":"#f44336","label":"Red line (Stress Score)",
             "text":"Shows your stress score (0–10) each day. When this line goes UP, your stress was higher that day."},
            {"icon":"🔗","color":"#ce93d8","label":"The relationship",
             "text":corr_msg},
            {"icon":"📅","color":"#90caf9","label":"Your best sleep day",
             "text":f"{best_day} — you slept the most on this day. Check if your stress was lower the next day."},
            {"icon":"⚠️","color":"#f44336","label":"Your worst sleep day",
             "text":f"{worst_day} — you slept the least on this day. This likely pushed your stress score higher."},
            {"icon":"📊","color":"#90caf9","label":"Your average stress score",
             "text":f"{avg_stress}/10 overall. {'This is high — immediate lifestyle changes needed.' if avg_stress>=6 else 'This is moderate — small improvements will help.' if avg_stress>=4 else 'This is healthy — keep your current habits going!'}"},
        ])

    # ── Raw Table ────────────────────────────────────────────────────────────
    st.markdown('<p class="sec">📋 Your Daily Records</p>', unsafe_allow_html=True)
    st.dataframe(df[["date","sleep_hrs","screen_time","stress_score","stress_level","sleep_debt"]],
                 use_container_width=True)
    st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode(), "sleep vs stress.csv", "text/csv")

def page_model(user):
    st.markdown("## 🤖 ML Model Info")
    st.markdown(f"**Algorithm:** Random Forest (150 trees) &nbsp;|&nbsp; **Test Accuracy:** `{ACC*100:.1f}%`")
    st.markdown("**Features:** Sleep Hrs, Screen Time, Exercise, Caffeine, Age Group &nbsp;|&nbsp; **Target:** Low / Medium / High stress")

    imp = RF.feature_importances_
    fig = px.bar(x=imp, y=FEATS, orientation='h', title="Feature Importance — What Affects Your Stress the Most?",
                 color=imp, color_continuous_scale=["#1565c0","#42a5f5","#ef5350"], labels={"x":"Importance","y":""})
    fig.update_layout(paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a", font_color="white",
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Plain-English Result Analysis ────────────────────────────────────────
    st.markdown('<p class="sec">📖 What This Chart Means — In Simple Words</p>', unsafe_allow_html=True)

    imp_dict = dict(zip(FEATS, imp))
    top = max(imp_dict, key=imp_dict.get)

    # Generate plain-English explanation per feature
    analysis = [
        ("😴", "Sleep Hrs", imp_dict["Sleep Hrs"],
         f"Sleep hours contributes **{imp_dict['Sleep Hrs']*100:.0f}%** to your stress prediction — the biggest factor. "
         "This means the number of hours you sleep each night is the #1 thing that determines whether your stress is Low, Medium or High. "
         "**If you change nothing else, fixing your sleep will reduce your stress the most.**"),

        ("📱", "Screen Time", imp_dict["Screen Time"],
         f"Screen time contributes **{imp_dict['Screen Time']*100:.0f}%** to your stress prediction — the 2nd biggest factor. "
         "The more hours you spend on your phone or computer, the higher your stress score becomes. "
         "**Reducing screen time, especially at night, is the second most powerful change you can make.**"),

        ("🏃", "Exercise", imp_dict["Exercise"],
         f"Exercise contributes **{imp_dict['Exercise']*100:.0f}%** to your stress prediction. "
         "Though it looks small on the chart, exercising even once a day acts as a protective factor — "
         "it pulls your stress score down. **Not exercising makes your stress worse even if you sleep well.**"),

        ("☕", "Caffeine", imp_dict["Caffeine"],
         f"Caffeine contributes **{imp_dict['Caffeine']*100:.0f}%** to your stress prediction. "
         "It has a smaller effect than sleep or screen time, but consuming caffeine late in the day "
         "disrupts sleep quality — which then increases your stress score indirectly. "
         "**Avoid caffeine after 2 PM for best results.**"),

        ("👥", "Age Group", imp_dict["Age Group"],
         f"Age group contributes **{imp_dict['Age Group']*100:.0f}%** to your stress prediction. "
         "Different age groups have different recommended sleep hours, so the model adjusts its scoring "
         "based on whether you are a child, teenager, adult, or senior. "
         "**A teenager sleeping 7 hrs is more sleep-deprived than an adult sleeping 7 hrs.**"),
    ]

    for emoji, name, score_, text in analysis:
        bar_color = "#f44336" if score_ >= 0.4 else "#ff9800" if score_ >= 0.1 else "#42a5f5"
        bar_pct   = round(score_ * 100)
        st.markdown(f"""
<div style="background:#0d1b2a;border-radius:12px;padding:16px 20px;margin:10px 0;
            border-left:4px solid {bar_color}">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <span style="font-size:1.4rem">{emoji}</span>
    <span style="color:white;font-weight:700;font-size:1rem">{name}</span>
    <span style="background:{bar_color};color:white;border-radius:20px;
                 padding:2px 12px;font-size:.8rem;font-weight:700;margin-left:auto">
      {bar_pct}% influence
    </span>
  </div>
  <div style="background:#1a2332;border-radius:6px;height:8px;margin-bottom:10px">
    <div style="background:{bar_color};border-radius:6px;height:8px;width:{min(bar_pct*1.4,100)}%"></div>
  </div>
  <p style="color:#b0bec5;font-size:.88rem;line-height:1.7;margin:0">{text}</p>
</div>""", unsafe_allow_html=True)

    # Summary box
    st.markdown(f"""
<div style="background:#0a1f2a;border:1px solid #42a5f5;border-radius:12px;
            padding:18px 22px;margin-top:16px">
  <p style="color:#90caf9;font-weight:700;font-size:1rem;margin-bottom:8px">
    🎯 Key Takeaway for You
  </p>
  <p style="color:#b0bec5;font-size:.9rem;line-height:1.8;margin:0">
    The model has learned that <strong style="color:#f44336">Sleep Hours</strong> is the most powerful 
    predictor of your stress — responsible for nearly <strong>{imp_dict['Sleep Hrs']*100:.0f}%</strong> 
    of the prediction. This matches medical research: chronic sleep deprivation is the leading 
    lifestyle cause of elevated cortisol and stress. <br><br>
    <strong>Simple rule from the model:</strong> Sleep more → Stress drops. Screen less → Stress drops. 
    Exercise daily → Stress drops further. Everything else is secondary.
  </p>
</div>""", unsafe_allow_html=True)

    st.markdown("""### ⚙️ How the Model Works
1. Your inputs (sleep, screen, exercise, caffeine, age group) are encoded as numbers
2. 150 decision trees each independently vote — Low, Medium, or High
3. The majority vote becomes your **Stress Level**
4. A weighted probability gives your **Stress Score (0–10)**
5. Lifestyle risk screening **only activates at High stress** — to prevent false alarms""")

def page_guide(user):
    st.markdown("## 📚 Condition Reference Guide")
    st.markdown('<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:12px 18px;margin-bottom:16px;color:#78909c;font-size:.83rem">⚕️ Educational reference only. Not a substitute for medical advice.</div>',unsafe_allow_html=True)
    for ag,dl in DIS.items():
        st.markdown(f"### 👥 {ag}")
        for d in dl:
            with st.expander(f"📋 {d['name']}"):
                st.markdown(f'<div style="background:#0a1f0a;border-left:3px solid #4caf50;border-radius:6px;padding:8px 14px;margin-bottom:10px;color:#a5d6a7;font-size:.82rem">📖 {d["ev"]}</div>',unsafe_allow_html=True)
                cc1,cc2=st.columns(2)
                with cc1:
                    st.markdown("**Symptoms:**")
                    for s in d['sym']: st.markdown(f"- {s}")
                with cc2:
                    st.markdown("**Treatment:**")
                    for t in d['tx']: st.markdown(f"- {t}")

def page_feedback_summary():
    st.markdown("## 💬 User Feedback Summary")
    df = get_all_feedback()
    if df.empty:
        st.info("No feedback submitted yet. Be the first!"); return

    total = len(df)
    avg_r = round(df['rating'].mean(), 1)
    stars = "⭐" * round(avg_r)

    # Summary metrics
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="mcard"><div class="mval" style="color:#ffd700">{stars}</div><div class="mlbl">Avg Rating ({avg_r}/5)</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mcard"><div class="mval" style="color:#42a5f5">{total}</div><div class="mlbl">Total Responses</div></div>', unsafe_allow_html=True)
    rec_yes = len(df[df['recommend'].str.contains("yes", case=False, na=False)])
    c3.markdown(f'<div class="mcard"><div class="mval" style="color:#4caf50">{round(rec_yes/total*100)}%</div><div class="mlbl">Would Recommend</div></div>', unsafe_allow_html=True)

    # Rating pie chart
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="rating", title="Rating Distribution",
                     color_discrete_sequence=["#f44336","#ff9800","#ffd700","#42a5f5","#4caf50"])
        fig.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(df, names="helpful", title="Was It Helpful?",
                      color_discrete_sequence=["#4caf50","#42a5f5","#ff9800","#f44336"])
        fig2.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    # Improvement suggestions bar chart
    if df['improve'].str.len().sum() > 0:
        all_tags = []
        for row in df['improve'].dropna():
            all_tags.extend([t.strip() for t in row.split(",") if t.strip() and t.strip() != "None"])
        if all_tags:
            tag_df = pd.Series(all_tags).value_counts().reset_index()
            tag_df.columns = ["Suggestion","Count"]
            fig3 = px.bar(tag_df, x="Count", y="Suggestion", orientation='h',
                          title="Most Requested Improvements",
                          color="Count", color_continuous_scale=["#1565c0","#42a5f5","#ef5350"])
            fig3.update_layout(paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                               font_color="white", coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)

    # Recent comments
    st.markdown('<p class="sec">📝 Recent Comments</p>', unsafe_allow_html=True)
    comments = df[df['comment'].str.len() > 0][['date','username','rating','comment']].head(10)
    for _, row in comments.iterrows():
        stars_r = "⭐" * int(row['rating'])
        st.markdown(f"""
<div style="background:#0d1b2a;border-left:4px solid #42a5f5;border-radius:8px;
     padding:12px 18px;margin:8px 0">
  <div style="color:#78909c;font-size:.8rem">{row['date']} &nbsp;|&nbsp; {row['username']} &nbsp;|&nbsp; {stars_r}</div>
  <div style="color:#e0e0e0;margin-top:6px;font-size:.9rem">"{row['comment']}"</div>
</div>""", unsafe_allow_html=True)

    st.download_button("⬇️ Export Feedback CSV", df.to_csv(index=False).encode(),
                       "feedback.csv", "text/csv")

def page_logout(user):
    # ── Goodbye banner ───────────────────────────────────────────────────────
    lv  = user.get('lv') or "Medium"
    sl  = user.get('sl') or REC_SL.get(user['ag'], 8)
    h, b = goodbye_msg(lv, sl, user['ag'], user['hr'])
    rec  = REC_SL.get(user['ag'], 8)
    now  = datetime.now()

    st.markdown(f"""<div class="bye">
    <h2>👋 Thanks for visiting, {user['u']}!</h2>
    <p style="color:#90caf9;font-size:1.1rem;font-weight:600">{h}</p>
    <p style="color:#b0bec5;line-height:1.8">{b}</p>
    <p style="color:#546e7a;font-size:.85rem;margin-top:12px">🌙 Sleep Vs Stress — Sleep well. Live better.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="talert">⏰ To get {rec} hrs ({user['ag']}),
    sleep by <strong>{(7+24-rec)%24:02d}:00</strong> tonight.<br>
    {"🚨 Past 10 PM — sleep immediately after logout!" if now.hour>=22
     else "💡 You still have time to wind down properly."}</div>""",
    unsafe_allow_html=True)

    st.markdown("---")

    # ── Feedback Form ────────────────────────────────────────────────────────
    st.markdown("## 💬 Share Your Feedback")
    st.markdown("*Before you go — tell us what you think! It takes 30 seconds.* 🙏")

    # Check if user already submitted today
    c = sqlite3.connect(DB)
    already = c.execute(
        "SELECT id FROM feedback WHERE username=? AND date=?",
        (user['u'], str(date.today()))).fetchone()
    c.close()

    if already:
        st.success("✅ You already submitted feedback today. Thank you!")
        # Show their submitted feedback
        c2 = sqlite3.connect(DB)
        row = c2.execute(
            "SELECT rating,comment FROM feedback WHERE username=? AND date=? ORDER BY id DESC LIMIT 1",
            (user['u'], str(date.today()))).fetchone()
        c2.close()
        if row:
            st.markdown(f"""
<div style="background:#0d1b2a;border-left:4px solid #4caf50;border-radius:8px;
     padding:14px 18px;margin:10px 0">
  <div style="color:#4caf50;font-weight:700">Your feedback: {"⭐"*int(row[0])}</div>
  <div style="color:#b0bec5;margin-top:6px">"{row[1]}"</div>
</div>""", unsafe_allow_html=True)
    else:
        with st.form("feedback_form", clear_on_submit=True):
            rating = st.select_slider(
                "⭐ How would you rate Sleep Vs Stress?",
                options=[1, 2, 3, 4, 5],
                value=4,
                format_func=lambda x: f"{'⭐'*x} ({x}/5)")

            helpful = st.radio(
                "💡 Did the stress analysis help you?",
                ["Yes, very helpful!", "Somewhat helpful", "Not really", "Not sure yet"],
                horizontal=True)

            improve = st.multiselect(
                "🔧 What would you like us to improve?",
                ["More diseases covered", "Better recommendations",
                 "Easier to use", "Faster loading", "More charts",
                 "Better sound alerts", "More age groups", "Nothing — it's great!"])

            recommend = st.radio(
                "🤝 Would you recommend Sleep Vs Stress to others?",
                ["Yes, definitely!", "Maybe", "No"], horizontal=True)

            comment = st.text_area(
                "📝 Any other comments or suggestions?",
                placeholder="Tell us what you think...", max_chars=300)

            submitted = st.form_submit_button("📤 Submit Feedback")

            if submitted:
                save_feedback({
                    "u": user['u'],
                    "dt": str(date.today()),
                    "rating": rating,
                    "helpful": helpful,
                    "improve": ", ".join(improve) if improve else "None",
                    "recommend": recommend,
                    "comment": comment
                })
                st.success("✅ Thank you for your feedback! It helps us improve Sleep Vs Stress.")
                st.balloons()

    st.markdown("---")

    # ── Feedback Summary ─────────────────────────────────────────────────────
    st.markdown("## 📊 What Others Are Saying")
    df_fb = get_all_feedback()

    if df_fb.empty:
        st.info("No feedback yet — be the first to submit above!")
    else:
        total = len(df_fb)
        avg_r = round(df_fb['rating'].mean(), 1)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="mcard"><div class="mval" style="color:#ffd700">{"⭐"*round(avg_r)}</div><div class="mlbl">Avg Rating ({avg_r}/5)</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="mcard"><div class="mval" style="color:#42a5f5">{total}</div><div class="mlbl">Total Reviews</div></div>', unsafe_allow_html=True)
        rec_yes = len(df_fb[df_fb['recommend'].str.contains("Yes", case=False, na=False)])
        c3.markdown(f'<div class="mcard"><div class="mval" style="color:#4caf50">{round(rec_yes/total*100)}%</div><div class="mlbl">Would Recommend</div></div>', unsafe_allow_html=True)

        # Rating pie
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df_fb, names="rating", title="Rating Distribution",
                         color_discrete_sequence=["#f44336","#ff9800","#ffd700","#42a5f5","#4caf50"])
            fig.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.pie(df_fb, names="helpful", title="Was It Helpful?",
                          color_discrete_sequence=["#4caf50","#42a5f5","#ff9800","#f44336"])
            fig2.update_layout(paper_bgcolor="#0d1b2a", font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        # Recent comments
        st.markdown('<p class="sec">📝 Recent Comments</p>', unsafe_allow_html=True)
        comments = df_fb[df_fb['comment'].str.len() > 0][['date','username','rating','comment']].head(5)
        for _, row in comments.iterrows():
            st.markdown(f"""
<div style="background:#0d1b2a;border-left:4px solid #42a5f5;border-radius:8px;
     padding:12px 18px;margin:8px 0">
  <div style="color:#78909c;font-size:.8rem">{row['date']} &nbsp;|&nbsp;
  {row['username']} &nbsp;|&nbsp; {"⭐"*int(row['rating'])}</div>
  <div style="color:#e0e0e0;margin-top:6px;font-size:.9rem">"{row['comment']}"</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    # ── Logout Button ────────────────────────────────────────────────────────
    st.markdown("### Ready to leave?")
    if st.button("🚪 Confirm Logout"):
        del st.session_state.user
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
PAGES = ["📊 Analyze","📈 History","🤖 Model Info","📚 Guide","🚪 Logout"]
PAGE_FN = {"📊 Analyze":page_analyze,"📈 History":page_history,
           "🤖 Model Info":page_model,"📚 Guide":page_guide,"🚪 Logout":page_logout}

def nav_buttons(current_page):
    """Renders Back / Next buttons at the bottom of every page."""
    idx = PAGES.index(current_page)
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if idx > 0:
            if st.button(f"⬅️ Back — {PAGES[idx-1].split(' ',1)[1]}",
                         use_container_width=True):
                st.session_state.current_page = PAGES[idx-1]
                st.rerun()
    with c2:
        st.markdown(
            f"<p style='text-align:center;color:#546e7a;font-size:.8rem;margin-top:10px'>"
            f"Page {idx+1} of {len(PAGES)}</p>",
            unsafe_allow_html=True)
    with c3:
        if idx < len(PAGES) - 1:
            if st.button(f"Next — {PAGES[idx+1].split(' ',1)[1]} ➡️",
                         use_container_width=True):
                st.session_state.current_page = PAGES[idx+1]
                st.rerun()

init_db()
if "user" not in st.session_state:
    page_login()
else:
    user = st.session_state.user

    # Initialise page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGES[0]

    # Sidebar radio — keep in sync with current_page
    st.sidebar.markdown(
        f"### 👤 {user['u']}\n**{user['ag']}** | Age {user['age']}\nLogin: {user['hr']:02d}:xx")
    selected = st.sidebar.radio("Navigate", PAGES,
                                index=PAGES.index(st.session_state.current_page))

    # If sidebar selection changed, update current_page
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()

    pg = st.session_state.current_page

    # Render the current page
    PAGE_FN[pg](user)

    # Show Next / Back buttons at the bottom of every page
    nav_buttons(pg)
