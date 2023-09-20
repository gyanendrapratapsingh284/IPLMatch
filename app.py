import streamlit as st
import pickle
import pandas as pd
import scikit-learn
st.title("IPL Win Predictor")

col1,col2 = st.columns(2)
teams = ['Sunrisers Hyderabad',
'Mumbai Indians',
'Royal Challengers Bangalore',
'Kolkata Knight Riders',
'Kings XI Punjab',
'Chennai Super Kings',
'Rajasthan Royals',
'Delhi Capitals']
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
'Sharjah', 'Mohali', 'Bengaluru']
pipe = pickle.load(open('pipe.pkl','rb'))
with col1:
    batting_team = st.selectbox("Select the batting Team",teams)

with col2:
    bowling_team = st.selectbox('Select the bowling Team',teams)
selected_city = st.selectbox("Select Host City",cities)
target = st.number_input("Target")
col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input("Score")
    if(score>300):
        st.warning("Are you sure this is correct score")
with col4:
    overs = st.number_input('Overs Completed')
    if(overs>=20):
        st.warning("Invalid Input")
with col5:
    wickets = st.number_input("Wickets out")
    if wickets>=10:
        st.warning("Invalid Inputs")

if st.button("Predicted Probability"):
    if (score<=300) and (overs<20) and (wickets<10):
        runs_left = target - score
        balls_left = 120-(overs*6)
        wickets = 10-wickets
        crr = score/overs
        rrr = (runs_left*6)/balls_left

        input_df = pd.DataFrame({'batting_team' : [batting_team],'bowling_team' : [bowling_team],'city':[selected_city],'runs_left':[runs_left],
                            'balls_left':[balls_left]
                            ,'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
        # st.table(input_df)
        result = pipe.predict_proba(input_df)
        win = result[0][0]
        loss = result[0][1]
        st.subheader(batting_team + " - " + str(round(loss*100)) + "%")
        st.subheader(bowling_team + " - " + str(round(win*100)) + "%")
