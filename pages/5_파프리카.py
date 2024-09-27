# 참외 출하량 예측 및 재학습

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_extras.app_logo import add_logo
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import save, load, set_random_seed
from dateutil import parser
from PIL import Image
from io import StringIO
from tkinter.tix import COLUMN
from pyparsing import empty
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import xlsxwriter
import datetime
import warnings
import pickle
import time
import io
import os
set_log_level("ERROR")
set_random_seed(0)
st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1.0,0.3])
image = Image.open('logo.png')
st.sidebar.image(image, caption='(주)빅데이터랩스', use_column_width=True)
st.sidebar.title('인공지능 기반 농작물 출하량 예측')
tab1, tab2 = st.tabs(['파프리카 출하량 예측', '파프리카 모델 재학습'])

with tab1:
    st.header('파프리카 출하량 예측')
    data = pd.read_excel('pages/yellowpaprika_train_data.xlsx', sheet_name=0)
    st.write("현재 파프리카 데이터 확인", data)
    st.markdown('날짜를 입력하고 **:blue[예측버튼]**을 클릭하여 이후 경매물량이 예측됩니다.')
    with st.form('Area-form'):
        d = st.date_input("언제부터 경매물량을 예측하시겠습니까?", datetime.date.today())
        st.write('선택한 날짜:', d)
        submitted = st.form_submit_button("예측하기")
        if submitted:
            st.write('예측결과:')    
            warnings.filterwarnings("ignore")
            with st.spinner('모델 로딩 중입니다.'):
                m_neural_prophet = load("pages/yellowpaprika_neuralprophet_model.np") #모델 로드
                time.sleep(5)
            st.success('예측 완료!')
            days = 365
            future_np = m_neural_prophet.make_future_dataframe(data, periods = days, n_historic_predictions=len(data)) 
            forecast_future = m_neural_prophet.predict(future_np)
            forecast_value = m_neural_prophet.get_latest_forecast(forecast_future, include_history_data=False)
            forecast_value.fillna(0,inplace=True)
            forecast_value['origin-0'] = forecast_value['origin-0'].apply(np.int64)
            forecast_value['origin-0'][forecast_value['origin-0']<0]=0
            forecast_value = forecast_value[['ds','origin-0']]
            np_data = np.array(forecast_value) # 수정 데이터
            week_weight = pd.read_excel('pages/적용할_노랑파프리카_요일가중치.xlsx', sheet_name=0)
            week_weight = np.array(week_weight)
            print(week_weight)
            start_date = pd.to_datetime(np_data[0, 0])
            end_date = pd.to_datetime(np_data[len(np_data)-1, 0])
            start_day_of_week = start_date.weekday()
            end_day_of_week = end_date.weekday()
            start_surplus_list = []
            end_surplus_list = []
            for i in range(6):
                if start_day_of_week == (6 - i):  
                    start_surplus_list = np_data[0:(i + 1), 1]
                    np_data = np_data[(i + 1):]
                    break  
            for i in range(6):
                if end_day_of_week == (i): 
                    end_surplus_list = np_data[-(i+1):, 1]
                    np_data = np_data[:-(i+1)]
                    break  
            weekend_sum_list = []
            sum = 0 
            count = 0 
            for i in range(len(np_data)):
                count += 1
                sum += np_data[i, 1]
                if count % 7 == 0:
                    weekend_sum_list.append(sum)
                    sum = 0
            weekend_sum_list = np.array(weekend_sum_list)
            result_list = []
            for k in range(len(weekend_sum_list)):
                for w in range(7): 
                    result_list.append((weekend_sum_list[k] * week_weight[0,w+1]).round(0))
            result_list = pd.DataFrame(result_list, columns=["가중치 적용 결과"])
            for i in range(len(start_surplus_list)):
                if i != len(start_surplus_list)-1:
                    start_surplus_list[i] = start_surplus_list[i] + start_surplus_list[len(start_surplus_list)-1]/(len(start_surplus_list)-1)
                    start_surplus_list[i] = np.around(start_surplus_list[i],0)
                else :
                    start_surplus_list[i] = 0
            start_surplus_list = pd.DataFrame(start_surplus_list, columns=["가중치 적용 결과"])
            end_surplus_list = pd.DataFrame(end_surplus_list, columns=["가중치 적용 결과"])
            weekend_weight_fin = pd.concat([start_surplus_list, result_list, end_surplus_list], axis=0).reset_index(drop=True)
            weekend_weight_fin = pd.concat([forecast_value.iloc[:,0:1], weekend_weight_fin], axis=1)
            weekend_weight_fin.columns = forecast_value.columns.tolist()
            weekend_weight_fin['origin-0'] = weekend_weight_fin['origin-0'].astype(int)
            forecast_value = weekend_weight_fin
            forecast_value= forecast_value[['ds','origin-0']].rename(columns={'ds':'날짜','origin-0':'예측경매물량_Neural Prophet'})
            model_scores = pd.read_excel('pages/yellowpaprika_neuralprophet_model_MAE_RMSE.xlsx', sheet_name=0)
            st.write("사용된 모델의 성능 - (현재 성능 개선 중)")
            st.write(f"MAE: {model_scores.iloc[0,0]}")
            st.write(f"RMSE: {model_scores.iloc[1,0]}")
            # st.write(f"결정계수: {model_scores.iloc[2,0]}")
            # st.write(f"수정된 결정계수: {model_scores.iloc[3,0]}")
            st.write("")
            st.write("")
            st.write("예측된 데이터의 추세 그래프")
            start_date = pd.Timestamp(d)
            end_date = start_date + pd.DateOffset(days=31)
            filtered_forecast_value = forecast_value[(forecast_value['날짜'] >= start_date) & (forecast_value['날짜'] <= end_date)] # 날짜 범위에 해당하는 데이터 필터링
            st.write(filtered_forecast_value ,use_container_width=True)
            st.write("예측된 데이터의 추세 그래프")
            fig_forecast = px.line(forecast_value, x='날짜', y='예측경매물량_Neural Prophet')
            fig_forecast.update_xaxes(range=[d, d + datetime.timedelta(days=31)])
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.write("실제 출하량의 추세 그래프")
            data_real_value = data[['ds', 'y']].rename(columns={'ds': '날짜', 'y': '실제 출하량'})
            fig_real = px.line(data_real_value, x='날짜', y='실제 출하량')
            st.plotly_chart(fig_real, use_container_width=True)

with tab2:
    st.header('파프리카 모델 재학습')
    st.title('모델 재학습')
    st.markdown('#### :white_check_mark: 데이터 양식')
    st.markdown('##### 데이터 양식은 엑셀 형식(.xlsx)으로 되어야 합니다.')
    st.markdown('##### 과거의 데이터가 아닌 새로 추가하고 싶은 데이터만 업로드 합니다.')
    st.markdown("##### 새로 추가하는 데이터의 변수는 다음과 같으며, 해당 변수의 이름은 영어로 작성합니다.")
    st.markdown("- 날짜 (ds)")
    st.markdown("- 경매물량 (y)")
    st.markdown("- 거래금액 (money)")    
    st.markdown("- 일요일 (sun_day)")
    st.markdown("")
    st.markdown("")
    st.markdown('#### :white_check_mark: 학습 데이터')
    old_data = pd.read_excel('pages/yellowpaprika_train_data.xlsx', sheet_name=0)
    st.markdown("##### 현재 사용되고 있는 학습 데이터")
    st.dataframe(old_data)
    st.markdown("")
    st.markdown("")
    st.markdown('#### :white_check_mark: 재학습을 위한 데이터 업로드')
    st.markdown('##### 새로운 데이터(.xlsx)를 불러온 다음 :blue[확인]버튼을 클릭하여 파일을 업로드합니다.')
    st.markdown('##### 업로드가 완료되면 :blue[확인]버튼을 클릭하여 재학습을 진행 합니다.')
        
    with st.form('학습데이터 업로드'):
        st.markdown('#### 데이터 업로드')
        uploaded_file = st.file_uploader("파일 선택")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8", 'ignore'))
            string_data = stringio.read()
            dataframe = pd.read_excel(uploaded_file) # 불러온 데이터를 데이터프레임 형식으로 변환
            st.markdown('#### 불러온 데이터')
            st.write(dataframe) # 불러온 데이터 출력
            st.markdown('#### :white_check_mark: 데이터 업로드 완료')
            st.markdown('확인을 누르면 학습이 시작됩니다.')
        submitted = st.form_submit_button("확인")
        if submitted: # 확인이 눌렸을때 
                dataframe = pd.concat([old_data, dataframe], axis=0)
                st.markdown('#### 새로 만들어진 학습 데이터')
                st.write(dataframe) # 새로 합쳐진 데이터 출력
                with st.spinner('Neural Prophet 학습중입니다.'):
                    forcast_len = 365
                    train_data = dataframe.iloc[:-forcast_len, :]
                    test_data = dataframe.iloc[-forcast_len:, :]
                    set_log_level("ERROR")
                    warnings.filterwarnings("ignore")
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    set_random_seed(0)
                    col_lst = ['money', 'sun_day'] 
                    m_neuralprophet = NeuralProphet(
                        n_forecasts = forcast_len,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        learning_rate=0.01,
                        epochs=100,
                        n_lags=7  
                    )
                    m_neuralprophet = m_neuralprophet.add_lagged_regressor(names=col_lst)
                    m_neuralprophet.fit(train_data, freq="D")
                    prediction_test = m_neuralprophet.make_future_dataframe(train_data, periods=forcast_len, n_historic_predictions=len(train_data))
                    forecast_future = m_neuralprophet.predict(prediction_test)
                    forecast_value = m_neuralprophet.get_latest_forecast(forecast_future, include_history_data=False)
                    forecast_value['origin-0'] = forecast_value['origin-0'].apply(np.int64) # 'origin-0' 열의 데이터를 정수형으로 변환
                    forecast_value['origin-0'][forecast_value['origin-0'] <0]=0 # 'origin-0' 열의 값이 0보다 작은 경우 이를 0으로 설정]
                    forecast_value.fillna(0, inplace=True) # 결측값을 0으로 대치
                    forecast_value = forecast_value[['ds','origin-0']]
                    np_data = np.array(forecast_value) # 수정 데이터
                    week_weight = pd.read_excel('pages/적용할_노랑파프리카_요일가중치.xlsx', sheet_name=0)
                    week_weight = np.array(week_weight)
                    start_date = pd.to_datetime(np_data[0, 0])
                    end_date = pd.to_datetime(np_data[len(np_data)-1, 0])
                    start_day_of_week = start_date.weekday()
                    end_day_of_week = end_date.weekday()
                    start_surplus_list = []
                    end_surplus_list = []
                    for i in range(6):
                        if start_day_of_week == (6 - i):  
                            start_surplus_list = np_data[0:(i + 1), 1]
                            np_data = np_data[(i + 1):]
                            break  
                    for i in range(6):
                        if end_day_of_week == (i):  
                            end_surplus_list = np_data[-(i+1):, 1]
                            np_data = np_data[:-(i+1)]
                            break  
                    weekend_sum_list = []
                    sum = 0 
                    count = 0 
                    for i in range(len(np_data)):
                        count += 1
                        sum += np_data[i, 1]
                        if count % 7 == 0:
                            weekend_sum_list.append(sum)
                            sum = 0
                    weekend_sum_list = np.array(weekend_sum_list)
                    result_list = []
                    for k in range(len(weekend_sum_list)):
                        for w in range(7): 
                            result_list.append((weekend_sum_list[k] * week_weight[0,w+1]).round(0))
                    result_list = pd.DataFrame(result_list, columns=["가중치 적용 결과"])
                    for i in range(len(start_surplus_list)):
                        if i != len(start_surplus_list)-1:
                            start_surplus_list[i] = start_surplus_list[i] + start_surplus_list[len(start_surplus_list)-1]/(len(start_surplus_list)-1)
                            start_surplus_list[i] = np.around(start_surplus_list[i],0)
                        else :
                            start_surplus_list[i] = 0
                    start_surplus_list = pd.DataFrame(start_surplus_list, columns=["가중치 적용 결과"])
                    end_surplus_list = pd.DataFrame(end_surplus_list, columns=["가중치 적용 결과"])
                    weekend_weight_fin = pd.concat([start_surplus_list, result_list, end_surplus_list], axis=0).reset_index(drop=True)
                    weekend_weight_fin = pd.concat([forecast_value.iloc[:,0:1], weekend_weight_fin], axis=1)
                    weekend_weight_fin.columns = forecast_value.columns.tolist()
                    weekend_weight_fin['origin-0'] = weekend_weight_fin['origin-0'].astype(int)
                    forecast_value = weekend_weight_fin        
                    compare_data = pd.merge(forecast_value[['ds','origin-0']], test_data[['ds','y']], how='outer',on='ds')
                    mae = mean_absolute_error(compare_data['y'], compare_data['origin-0'])
                    rmse = np.sqrt(mean_squared_error(compare_data['y'], compare_data['origin-0']))   
                    R2 = r2_score(compare_data['y'], compare_data['origin-0'])
                    adj_r2 = 1 - (1 - R2) * (forcast_len - 1) / (forcast_len - len(col_lst) - 1)                   
                    st.write("모델 성능")
                    st.write(f"MAE: {mae}")
                    st.write(f"RMSE: {rmse}")
                    st.write(f"결정계수: {R2}")
                    st.write(f"수정된 결정계수: {adj_r2}")
                    list = [mae, rmse, R2, adj_r2]
                    df = pd.DataFrame(list)
                    df.to_excel('pages/yellowpaprika_neuralprophet_model_MAE_RMSE.xlsx', index=False) # 파일로 저장
                    real_m_neuralprophet = NeuralProphet(
                        n_forecasts = forcast_len,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        learning_rate=0.01,
                        epochs=100,
                        n_lags=7 
                    )
                    real_m_neuralprophet = real_m_neuralprophet.add_lagged_regressor(names=col_lst)
                    real_m_neuralprophet.fit(dataframe, freq="D")
                    save(real_m_neuralprophet, "pages/yellowpaprika_neuralprophet_model.np")
                    time.sleep(5)
                    st.success(':white_check_mark: Neural Prophet 모델 학습완료')
