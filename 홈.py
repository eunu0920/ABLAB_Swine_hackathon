from PIL import Image
import streamlit as st
from tkinter.tix import COLUMN
from pyparsing import empty

st.set_page_config(
    page_title="AI 기반 농작물 출하량 예측", 
    page_icon="👋", 
    layout="wide"
)
empty1,con1,empty2 = st.columns([0.1,1.0,0.1])
empyt1,con2,con3,con4,con5,empty2 = st.columns([0.1,0.25,0.25,0.25,0.25,0.1])
empyt1,con6,empty2 = st.columns([0.1,1.0,0.1])
image = Image.open('logo.png')
st.sidebar.image(image, caption='(주)빅데이터랩스', use_column_width=True)
st.sidebar.title('인공지능 기반 농작물 출하량 예측')

with con1:
    st.title('인공지능 기반 농작물 출하량 예측')
    st.text('메뉴 설명')
    st.text('1. 홈 : 메뉴 설명 및 웹페이지 설명')
    st.text('2. 참외 : 3단 출하량을 기준으로 계산하며, 사용자가 입력한 날짜로 부터 1달간 참외 생산량 예측,')
    st.text('3. 오이 : 사용자가 입력한 날짜로 부터 1달간 오이 생산량 예측')
    st.text('4. 감귤 : 사용자가 입력한 날짜로 부터 1달간 감귤 생산량 예측')
    st.text('5. 파프리카 : 사용자가 입력한 날짜로 부터 1달간 파프리카 생산량 예측')
    st.text("")
    st.text("")
    st.text("")

with con2 :
    image_2 = Image.open('참외.png')
    st.image(image_2, caption='참외', use_column_width=True)

with con3 :
    image_3 = Image.open('오이.png')
    st.image(image_3, caption='오이', use_column_width=True)

with con4 :
    image_4 = Image.open('하우스감귤.png')
    st.image(image_4, caption='하우스감귤', use_column_width=True)

with con5 :
    image_5 = Image.open('파프리카.png')
    st.image(image_5, caption='파프리카', use_column_width=True)
    
with con6 :
    st.subheader('이 웹사이트는 최적의 :blue[Neural Prophet] 모델을 사용하여 농작물의 생산량을 예측합니다.')
    st.text("각 농장물에 대한 갱신은 한 달에 한번씩 진행되며, 일자는 추후에 제공합니다.")
    st.text("최적 모델의 경우 더 나은 모델이 생성되는 즉시 갱신됨을 알립니다.")
    st.text("")
    st.text("")

