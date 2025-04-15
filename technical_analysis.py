## AI-Powered Technical Analysis Dashboard (Gemini 2.0)
## Source: https://www.youtube.com/@DeepCharts

# Libraries
import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# 기술적 지표 계산 함수들
def calculate_sma(data, window=20):
    """단순 이동평균 계산"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span=20):
    """지수 이동평균 계산"""
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    """볼린저 밴드 계산"""
    sma = calculate_sma(data, window)
    stddev = data['Close'].rolling(window=window).std()
    upper_band = sma + (stddev * num_std)
    lower_band = sma - (stddev * num_std)
    return upper_band, sma, lower_band

def calculate_vwap(data):
    """거래량 가중 평균가격 계산"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    volume = data['Volume']
    vp = typical_price * volume
    cv = volume.cumsum()
    return (vp.cumsum() / cv)

def calculate_rsi(data, window=14):
    """상대강도지수(RSI) 계산"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_span=12, slow_span=26, signal_span=9):
    """이동평균수렴확산(MACD) 계산"""
    exp1 = data['Close'].ewm(span=fast_span, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_span, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_mfi(data, window=14):
    """Money Flow Index (MFI) 계산"""
    # 전형적인 가격 계산
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    # 자금 흐름 계산
    money_flow = typical_price * data['Volume']
    
    # 양수/음수 자금 흐름 초기화
    positive_flow = pd.Series(0.0, index=data.index)
    negative_flow = pd.Series(0.0, index=data.index)
    
    # 양수/음수 자금 흐름 계산
    for i in range(1, len(data)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = float(money_flow.iloc[i])
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = float(money_flow.iloc[i])
    
    # 14일 기간의 양수/음수 흐름 합계 계산
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    # MFI 계산
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def create_chart(data, ticker, indicators):
    """
    주어진 데이터와 지표를 사용하여 차트를 생성합니다.
    """
    try:
        # 오실레이터 지표 확인
        oscillator_indicators = [ind for ind in indicators if ind in ["RSI", "MACD", "MFI"]]
        price_indicators = [ind for ind in indicators if ind not in ["RSI", "MACD", "MFI"]]
        
        # 서브플롯 생성 (오실레이터 지표 수에 따라 행 수 결정)
        rows = 1 + len(oscillator_indicators)
        row_heights = [0.6] + [0.4/len(oscillator_indicators)] * len(oscillator_indicators) if oscillator_indicators else [1]
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=row_heights)
        
        # 캔들스틱 추가
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # 가격 기반 지표 추가
        for indicator in price_indicators:
            add_price_indicator(fig, data, indicator)
        
        # 오실레이터 지표 추가
        current_row = 2
        for indicator in oscillator_indicators:
            add_oscillator_indicator(fig, data, indicator, current_row)
            current_row += 1
        
        # 차트 레이아웃 설정
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f"{ticker} Stock Price Chart",
            height=200 * rows,  # 행 수에 따라 높이 조정
            template="plotly_white"
        )
        
        return fig
    
    except Exception as e:
        st.error(f"차트 생성 중 오류 발생: {str(e)}")
        return None

def add_price_indicator(fig, data, indicator, row=1, col=1):
    """가격 차트에 기술적 지표 추가"""
    if indicator == "SMA":
        sma20 = calculate_sma(data)
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='20-Day SMA', 
                                line=dict(color='blue')), row=row, col=col)
    
    elif indicator == "EMA":
        ema20 = calculate_ema(data)
        fig.add_trace(go.Scatter(x=data.index, y=ema20, name='20-Day EMA', 
                                line=dict(color='orange')), row=row, col=col)
    
    elif indicator == "Bollinger Bands":
        upper_band, sma20, lower_band = calculate_bollinger_bands(data)
        
        fig.add_trace(go.Scatter(x=data.index, y=upper_band, name='Upper Band', 
                                line=dict(color='rgba(250, 0, 0, 0.5)')), row=row, col=col)
        fig.add_trace(go.Scatter(x=data.index, y=sma20, name='20-Day SMA', 
                                line=dict(color='rgba(0, 0, 250, 0.5)')), row=row, col=col)
        fig.add_trace(go.Scatter(x=data.index, y=lower_band, name='Lower Band', 
                                line=dict(color='rgba(250, 0, 0, 0.5)')), row=row, col=col)
    
    elif indicator == "VWAP":
        vwap = calculate_vwap(data)
        fig.add_trace(go.Scatter(x=data.index, y=vwap, name='VWAP', 
                                line=dict(color='purple')), row=row, col=col)

def add_oscillator_indicator(fig, data, indicator, row, col=1):
    """오실레이터 지표를 별도의 서브플롯에 추가"""
    if indicator == "RSI":
        rsi = calculate_rsi(data)
        
        # RSI 추가
        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', 
                                line=dict(color='green')), row=row, col=col)
        
        # 과매수/과매도 영역 표시
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", 
                     opacity=0.5, row=row, col=col)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", 
                     opacity=0.5, row=row, col=col)
        
        # Y축 범위 설정
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=col)
    
    elif indicator == "MACD":
        macd, signal, histogram = calculate_macd(data)
        
        # MACD 라인 추가
        fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', 
                                line=dict(color='blue')), row=row, col=col)
        fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal', 
                                line=dict(color='red')), row=row, col=col)
        fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', 
                            marker_color='green'), row=row, col=col)
        
        # Y축 제목 설정
        fig.update_yaxes(title_text="MACD", row=row, col=col)
    
    elif indicator == "MFI":
        mfi = calculate_mfi(data)
        
        # MFI 추가
        fig.add_trace(go.Scatter(x=data.index, y=mfi, name='MFI', 
                                line=dict(color='purple')), row=row, col=col)
        
        # 과매수/과매도 영역 표시
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="red", 
                     opacity=0.5, row=row, col=col)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="green", 
                     opacity=0.5, row=row, col=col)
        
        # Y축 범위 설정
        fig.update_yaxes(title_text="MFI", range=[0, 100], row=row, col=col)

def display_chart(fig, analysis):
    """차트와 분석 결과를 화면에 표시합니다."""
    if fig is None:
        st.error("차트를 생성할 수 없습니다.")
        return
    
    try:
        # 차트 객체 타입 확인
        if not isinstance(fig, (go.Figure, make_subplots)):
            st.error(f"유효하지 않은 차트 객체 타입: {type(fig)}")
            return
            
        # 차트에 데이터가 있는지 확인
        if not fig.data or len(fig.data) == 0:
            st.error("차트에 표시할 데이터가 없습니다.")
            return
            
        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)
        
        # 분석 결과 표시
        if analysis and isinstance(analysis, dict):
            st.subheader("AI 분석 결과")
            if 'action' in analysis:
                st.write(f"**추천 액션:** {analysis['action']}")
            if 'justification' in analysis:
                st.write(f"**근거:** {analysis['justification']}")
        elif analysis:
            st.error(f"유효하지 않은 분석 결과 형식: {type(analysis)}")
    except Exception as e:
        st.error(f"차트 표시 중 오류 발생: {str(e)}")
        
        # 상세한 에러 정보 제공
        st.error("차트 표시 중 문제가 발생했습니다. 다음 정보를 확인해주세요:")
        st.error(f"- 에러 유형: {type(e).__name__}")
        st.error(f"- 에러 메시지: {str(e)}")
        
        # 디버깅 정보 제공
        if fig is not None:
            st.write("차트 데이터 정보:")
            st.write(f"- 차트 데이터 유형: {type(fig)}")
            st.write(f"- 차트 데이터 속성: {dir(fig)[:10]}...")

def setup_gemini_api():
    """Gemini API 설정"""
    try:
        GOOGLE_API_KEY = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=GOOGLE_API_KEY)
        MODEL_NAME = 'gemini-2.0-flash'  # or other model
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"API 키를 불러오는데 실패했습니다: {e}. .streamlit/secrets.toml 파일을 확인해주세요.")
        return None

def format_ticker_symbol(ticker, market):
    """시장에 따른 티커 형식 조정"""
    if market == "한국(KRX)" and ticker.isdigit() and len(ticker) == 6:
        # 이미 올바른 형식
        return ticker
    elif market == "지수(Indices)" and not ticker.startswith('^'):
        # 지수 심볼에 ^ 접두사 추가 (없는 경우)
        return f"^{ticker}"
    return ticker

def validate_data(data, ticker):
    """데이터 유효성 검사 및 필요한 열 확인"""
    if data is None or data.empty:
        st.error(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
        return None
    
    # 필요한 열이 있는지 확인
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 열 이름이 다를 경우 매핑 (대소문자 차이 등)
    column_mapping = {}
    for col in data.columns:
        for req_col in required_columns:
            if col.upper() == req_col.upper():
                column_mapping[col] = req_col
    
    # 열 이름 표준화
    if column_mapping:
        data = data.rename(columns=column_mapping)
    
    # 필요한 열이 모두 있는지 확인
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"{ticker}에 필요한 열이 없습니다: {', '.join(missing_cols)}")
        return None
    
    return data

def fetch_stock_data(ticker, start_date, end_date, market="미국(US)"):
    """주식 데이터 가져오기"""
    try:
        ticker_symbol = format_ticker_symbol(ticker, market)
        data = fdr.DataReader(ticker_symbol, start=start_date, end=end_date)
        return validate_data(data, ticker)
    except Exception as e:
        st.error(f"{ticker} 데이터 다운로드 중 오류 발생: {str(e)}")
        return None

def analyze_with_ai(data, ticker):
    """AI를 사용하여 기술적 분석 수행"""
    try:
        # 데이터가 비어있는지 확인
        if data.empty or len(data) < 2:
            return {
                "action": "데이터 부족", 
                "justification": "분석을 위한 충분한 데이터가 없습니다."
            }
        
        # 기본 기술적 지표 계산
        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        
        # 추가 기술적 분석을 위한 데이터 준비
        change_pct = ((last_close - prev_close) / prev_close) * 100
        
        # 단순 이동평균(SMA) 계산
        sma_20 = calculate_sma(data, window=20).iloc[-1]
        sma_50 = calculate_sma(data, window=50).iloc[-1]
        sma_200 = calculate_sma(data, window=200).iloc[-1]
        
        # RSI 계산
        rsi = calculate_rsi(data).iloc[-1]
        
        # MACD 계산
        macd, signal, histogram = calculate_macd(data)
        macd_last = macd.iloc[-1]
        signal_last = signal.iloc[-1]
        histogram_last = histogram.iloc[-1]
        
        # 볼린저 밴드 계산
        upper_band, middle_band, lower_band = calculate_bollinger_bands(data)
        upper_last = upper_band.iloc[-1]
        middle_last = middle_band.iloc[-1]
        lower_last = lower_band.iloc[-1]
        
        # 분석 시작
        details = []
        
        # 가격 동향 분석
        details.append(f"**가격 동향**: {ticker}의 최근 종가는 {last_close:.2f}로, 전일 대비 {change_pct:.2f}% {'상승' if change_pct > 0 else '하락'}했습니다.")
        
        # 이동평균 분석
        ma_analysis = []
        if last_close > sma_20:
            ma_analysis.append("현재 가격이 20일 이동평균선 위에 있어 단기적으로 강세를 보입니다.")
        else:
            ma_analysis.append("현재 가격이 20일 이동평균선 아래에 있어 단기적으로 약세를 보입니다.")
            
        if sma_20 > sma_50:
            ma_analysis.append("20일 이동평균선이 50일 이동평균선 위에 있어 중단기 상승 추세입니다.")
        else:
            ma_analysis.append("20일 이동평균선이 50일 이동평균선 아래에 있어 중단기 하락 추세입니다.")
            
        if last_close > sma_200:
            ma_analysis.append("현재 가격이 200일 이동평균선 위에 있어 장기적으로 강세를 보입니다.")
        else:
            ma_analysis.append("현재 가격이 200일 이동평균선 아래에 있어 장기적으로 약세를 보입니다.")
            
        details.append(f"**이동평균 분석**: {' '.join(ma_analysis)}")
        
        # RSI 분석
        rsi_status = "과매수(매도 신호)" if rsi > 70 else "과매도(매수 신호)" if rsi < 30 else "중립"
        details.append(f"**RSI 분석**: 현재 RSI 값은 {rsi:.2f}로, {rsi_status} 상태입니다.")
        
        # MACD 분석
        macd_status = ""
        if macd_last > signal_last and macd_last > 0:
            macd_status = "강한 매수 신호"
        elif macd_last > signal_last and macd_last < 0:
            macd_status = "약한 매수 신호"
        elif macd_last < signal_last and macd_last > 0:
            macd_status = "약한 매도 신호"
        else:
            macd_status = "강한 매도 신호"
            
        details.append(f"**MACD 분석**: MACD({macd_last:.2f})와 시그널({signal_last:.2f})의 관계는 {macd_status}를 나타냅니다.")
        
        # 볼린저 밴드 분석
        bb_position = ((last_close - lower_last) / (upper_last - lower_last)) * 100
        bb_status = ""
        if bb_position > 80:
            bb_status = "상단 밴드에 근접하여 과매수 가능성 있음"
        elif bb_position < 20:
            bb_status = "하단 밴드에 근접하여 과매도 가능성 있음"
        else:
            bb_status = "밴드 내에서 정상 거래 중"
            
        details.append(f"**볼린저 밴드 분석**: 현재 가격은 밴드 내에서 {bb_position:.1f}% 위치에 있으며, {bb_status}.")
        
        # 주요 추세 파악 및 요약
        trend_signals = []
        if last_close > sma_20 and sma_20 > sma_50:
            trend_signals.append("단기 상승세")
        if last_close > sma_200:
            trend_signals.append("장기 상승세")
        if rsi < 30:
            trend_signals.append("과매도")
        if rsi > 70:
            trend_signals.append("과매수")
        if macd_last > signal_last and histogram_last > 0:
            trend_signals.append("MACD 매수 신호")
        if bb_position < 20:
            trend_signals.append("볼린저 밴드 하단 지지")
        if bb_position > 80:
            trend_signals.append("볼린저 밴드 상단 저항")
            
        trend_summary = "중립적" if not trend_signals else ", ".join(trend_signals)
        details.append(f"**종합 시그널**: {trend_summary}")
        
        # 액션 결정
        # 여러 지표를 종합적으로 고려하여 액션 결정
        buy_signals = 0
        sell_signals = 0
        
        # 이동평균 신호
        if last_close > sma_20:
            buy_signals += 1
        else:
            sell_signals += 1
            
        if sma_20 > sma_50:
            buy_signals += 1
        else:
            sell_signals += 1
            
        # RSI 신호
        if rsi < 30:
            buy_signals += 2  # 강한 매수 신호
        elif rsi < 45:
            buy_signals += 1
        elif rsi > 70:
            sell_signals += 2  # 강한 매도 신호
        elif rsi > 55:
            sell_signals += 1
            
        # MACD 신호
        if macd_last > signal_last:
            buy_signals += 1
        else:
            sell_signals += 1
            
        # 볼린저 밴드 신호
        if bb_position < 20:
            buy_signals += 1
        elif bb_position > 80:
            sell_signals += 1
            
        # 최종 액션 결정
        if buy_signals > sell_signals + 2:
            action = "적극 매수 고려"
        elif buy_signals > sell_signals:
            action = "매수 고려"
        elif sell_signals > buy_signals + 2:
            action = "적극 매도 고려"
        elif sell_signals > buy_signals:
            action = "매도 고려"
        else:
            action = "관망 추천"
            
        # 주요 근거 요약
        justification = f"{ticker}의 기술적 지표 분석 결과, 매수 시그널 {buy_signals}개와 매도 시그널 {sell_signals}개가 감지되었습니다."
        
        # 최종 결과 반환
        return {
            "action": action, 
            "justification": justification, 
            "details": "\n\n".join(details)
        }
    except Exception as e:
        return {
            "action": "분석 오류", 
            "justification": f"AI 분석 중 오류 발생: {str(e)}",
            "details": f"분석 과정에서 다음 오류가 발생했습니다: {str(e)}\n\n오류 유형: {type(e).__name__}"
        }

def analyze_ticker_with_data(data, ticker, indicators, use_ai_analysis=False):
    """이미 다운로드된 데이터를 사용하여 차트 생성 및 분석"""
    try:
        # 데이터 유효성 검사
        if data is None or data.empty:
            st.error(f"{ticker}에 대한 유효한 데이터가 없습니다.")
            return None
        
        # 차트 생성
        fig = create_chart(data, ticker, indicators)
        
        # 결과 딕셔너리 초기화
        result = {"chart": fig, "data": data}
        
        # AI 분석 수행 (선택 사항)
        if use_ai_analysis and fig is not None:
            with st.spinner(f"AI가 {ticker}를 분석 중입니다..."):
                result["ai_analysis"] = analyze_with_ai(data, ticker)
        
        return result
    
    except Exception as e:
        st.error(f"{ticker} 분석 중 오류 발생: {str(e)}")
        return None

def show_data_source_info():
    """데이터 소스 정보 표시"""
    with st.expander("데이터 소스 정보"):
        st.markdown("""
        이 대시보드는 **FinanceDataReader**를 사용하여 주식 데이터를 가져옵니다.
        
        **지원하는 티커 형식**:
        - **미국 주식**: 'AAPL', 'MSFT', 'GOOGL' 등 (심볼)
        - **한국 주식**: '005930', '035720' 등 (종목코드 6자리)
        - **지수**: 'KS11' (KOSPI), 'KQ11' (KOSDAQ), 'DJI' (다우존스), 'IXIC' (나스닥) 등
        
        **사용 예시**:
        - 삼성전자: '005930'
        - 카카오: '035720'
        - 애플: 'AAPL'
        - 코스피 지수: 'KS11'
        """)

def show_indicators_info():
    """기술적 지표 설명 표시"""
    with st.expander("기술적 지표 설명"):
        st.markdown("""
        **지원하는 기술적 지표**:
        - **SMA (단순 이동평균)**: 가격의 단순 이동평균을 표시합니다.
        - **EMA (지수 이동평균)**: 최근 데이터에 더 많은 가중치를 두는 이동평균입니다.
        - **Bollinger Bands (볼린저 밴드)**: 가격 변동성을 기반으로 한 상한선과 하한선을 표시합니다.
        - **RSI (상대강도지수)**: 가격 변동의 강도와 속도를 측정합니다.
        - **MACD (이동평균수렴확산)**: 두 이동평균선 간의 관계를 보여주는 추세 지표입니다.
        - **VWAP (거래량가중평균가격)**: 거래량을 고려한 평균 가격을 표시합니다.
        - **MFI (자금흐름지수)**: 가격과 거래량을 결합한 지표로, 과매수/과매도 상태를 파악합니다.
        """)

def setup_sidebar():
    """사이드바 설정"""
    st.sidebar.header("설정")
    
    # AI 분석 활성화 여부
    use_ai_analysis = st.sidebar.checkbox(
        "AI 분석 사용", 
        value=True, 
        help="체크하면 Gemini AI를 사용한 분석을 수행합니다. 체크하지 않으면 차트만 표시합니다."
    )
    
    # 시장 선택
    market = st.sidebar.selectbox(
        "시장 선택:",
        ["미국(US)", "한국(KRX)", "지수(Indices)"],
        help="분석할 주식이 속한 시장을 선택하세요."
    )
    
    # 티커 입력 방식 개선
    ticker_help = {
        "미국(US)": "예: AAPL, MSFT, GOOGL",
        "한국(KRX)": "예: 005930 (삼성전자), 035720 (카카오)",
        "지수(Indices)": "예: KS11 (KOSPI), KQ11 (KOSDAQ), DJI (다우존스)"
    }
    
    ticker_placeholder = {
        "미국(US)": "AAPL,MSFT,GOOGL",
        "한국(KRX)": "005930,035720",
        "지수(Indices)": "KS11,KQ11,DJI"
    }
    
    # Input for multiple stock tickers (comma-separated)
    tickers_input = st.sidebar.text_input(
        "티커 입력 (쉼표로 구분):", 
        ticker_placeholder[market],
        help=ticker_help[market]
    )
    
    # Parse tickers by stripping whitespace and splitting on commas
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") 
              if ticker.strip()]
    
    # Set the date range
    end_date_default = datetime.today()
    start_date_default = end_date_default - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", value=start_date_default)
    end_date = st.sidebar.date_input("End Date", value=end_date_default)
    
    # Technical indicators selection
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "VWAP", "MFI"],
        default=["SMA", "Bollinger Bands"]
    )
    
    return use_ai_analysis, market, tickers, start_date, end_date, indicators

def display_results(tickers, results):
    """분석 결과 표시"""
    # 탭 생성: 첫 번째 탭은 전체 요약, 이후 탭은 각 티커별
    tabs = st.tabs(["Summary"] + list(results.keys()))
    
    # 요약 탭
    with tabs[0]:
        st.header("분석 요약")
        if results:
            summary_data = []
            for ticker, result in results.items():
                if "ai_analysis" in result:
                    summary_data.append({
                        "티커": ticker,
                        "추천": result["ai_analysis"].get("action", "N/A"),
                        "근거": result["ai_analysis"].get("justification", "N/A")
                    })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data))
            else:
                st.info("AI 분석 결과가 없습니다.")
        else:
            st.info("분석 결과가 없습니다.")
    
    # 각 티커별 탭
    for i, ticker in enumerate(results.keys(), 1):
        with tabs[i]:
            st.header(f"{ticker} 분석")
            result = results[ticker]
            
            # 차트 표시
            if "chart" in result:
                st.plotly_chart(result["chart"], use_container_width=True)
            
            # AI 분석 결과 표시
            if "ai_analysis" in result:
                st.subheader("AI 분석 결과")
                
                # 컬러 표시로 추천 액션을 강조
                action = result['ai_analysis'].get('action', 'N/A')
                action_color = "green"
                if "매도" in action:
                    action_color = "red"
                elif "관망" in action:
                    action_color = "orange"
                
                # 액션과 근거 표시
                st.markdown(f"**추천 행동**: <span style='color:{action_color};font-weight:bold'>{action}</span>", unsafe_allow_html=True)
                st.write(f"**근거**: {result['ai_analysis'].get('justification', 'N/A')}")
                
                # 상세 분석 정보 표시
                with st.expander("상세 분석 정보", expanded=True):
                    details = result['ai_analysis'].get('details', '상세 정보가 없습니다.')
                    st.markdown(details)

def set_port():
    """Streamlit 앱의 포트를 3333으로 설정"""
    import os
    os.environ['STREAMLIT_SERVER_PORT'] = '3333'

def main():
    set_port()  # 포트 설정
    # 페이지 설정
    st.set_page_config(
        page_title="AI-Powered Technical Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    st.title("AI-Powered Technical Stock Analysis Dashboard")
    
    # 정보 표시
    show_data_source_info()
    show_indicators_info()
    
    # Gemini API 설정
    gen_model = setup_gemini_api()
    
    # 사이드바 설정
    use_ai_analysis, market, tickers, start_date, end_date, indicators = setup_sidebar()
    
    # 데이터 가져오기 버튼
    if st.sidebar.button("Fetch Data"):
        stock_data = {}
        for ticker in tickers:
            data = fetch_stock_data(ticker, start_date, end_date, market)
            if data is not None:
                stock_data[ticker] = data
        
        if stock_data:
            st.session_state["stock_data"] = stock_data
            tickers_loaded = ", ".join(stock_data.keys())
            st.success(f"Stock data loaded successfully for: {tickers_loaded}")
        else:
            st.error("데이터를 가져오지 못했습니다. 다른 티커나 날짜 범위를 시도해보세요.")
    
    # 분석 실행 버튼
    if st.sidebar.button("분석 실행"):
        if not tickers:
            st.error("분석할 티커를 입력해주세요.")
        else:
            results = {}
            
            # 세션 상태에 저장된 데이터 확인
            if "stock_data" in st.session_state and st.session_state["stock_data"]:
                stock_data = st.session_state["stock_data"]
                
                for ticker in tickers:
                    if ticker in stock_data:
                        # 저장된 데이터로 분석
                        result = analyze_ticker_with_data(
                            stock_data[ticker], ticker, indicators, use_ai_analysis
                        )
                    else:
                        # 새로 데이터 가져와서 분석
                        data = fetch_stock_data(ticker, start_date, end_date, market)
                        result = analyze_ticker_with_data(
                            data, ticker, indicators, use_ai_analysis
                        )
                    
                    if result:
                        results[ticker] = result
            else:
                # 모든 티커에 대해 새로 데이터 가져오기
                for ticker in tickers:
                    data = fetch_stock_data(ticker, start_date, end_date, market)
                    result = analyze_ticker_with_data(
                        data, ticker, indicators, use_ai_analysis
                    )
                    
                    if result:
                        results[ticker] = result
            
            # 결과 표시
            if results:
                display_results(tickers, results)
            else:
                st.warning("표시할 분석 결과가 없습니다.")
    
    # 저장된 데이터가 있으면 표시
    elif "stock_data" in st.session_state and st.session_state["stock_data"]:
        st.info("데이터가 로드되었습니다. '분석 실행' 버튼을 클릭하여 분석을 시작하세요.")
    else:
        st.info("분석을 시작하려면 티커를 입력하고 'Fetch Data' 버튼을 클릭하세요.")

if __name__ == "__main__":
    main()

# 이 앱은 .streamlit/config.toml 설정에 따라 http://localhost:8501 에서 실행됩니다.
# 실행 방법: streamlit run technical_analysis.py
