# 1d_KAMA_Trend_RSI14_StochasticFilter
# Hypothesis: On 1d timeframe, KAMA adapts to market noise, providing a robust trend filter.
# Combined with RSI(14) for momentum and Stochastic(14,3) for overbought/oversold conditions.
# In bull markets: go long when KAMA trends up, RSI > 50, and Stochastic not overbought.
# In bear markets: go short when KAMA trends down, RSI < 50, and Stochastic not oversold.
# Volatility filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid choppy markets.
# Position size: 0.25 to manage risk during large drawdowns.
# Target: 15-25 trades per year (~60-100 over 4 years) to minimize fee drag.

name = "1d_KAMA_Trend_RSI14_StochasticFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA (Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=1))
    abs_change = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(change) > 0 else 0
    # Vectorized ER calculation
    er = np.zeros(n)
    for i in range(er_len, n):
        if i >= er_len:
            change_period = np.abs(close[i-er_len+1:i+1] - close[i-er_len:i])
            sum_change = np.sum(change_period)
            if sum_change > 0:
                er[i] = np.abs(close[i] - close[i-er_len]) / sum_change
            else:
                er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = (np.sum(gain[1:i+1]) / i) if i > 0 else 0
            avg_loss[i] = (np.sum(loss[1:i+1]) / i) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3)
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    for i in range(n):
        if i < 13:
            lowest_low[i] = np.min(low[0:i+1]) if i >= 0 else low[i]
            highest_high[i] = np.max(high[0:i+1]) if i >= 0 else high[i]
        else:
            lowest_low[i] = np.min(low[i-13:i+1])
            highest_high[i] = np.max(high[i-13:i+1])
    stoch = np.where((highest_high - lowest_low) != 0, 
                     (close - lowest_low) / (highest_high - lowest_low) * 100, 
                     50)
    stoch_k = np.zeros(n)
    stoch_d = np.zeros(n)
    for i in range(n):
        if i < 2:
            stoch_k[i] = stoch[i]
        else:
            stoch_k[i] = np.mean(stoch[max(0, i-2):i+1])
    for i in range(n):
        if i < 2:
            stoch_d[i] = stoch_k[i]
        else:
            stoch_d[i] = np.mean(stoch_k[max(0, i-2):i+1])
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.zeros(n)
    atr_50 = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_14[i] = np.mean(tr[0:i+1]) if i >= 0 else tr[i]
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
        if i < 49:
            atr_50[i] = np.mean(tr[0:i+1]) if i >= 0 else tr[i]
        else:
            atr_50[i] = np.mean(tr[i-49:i+1])
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(stoch_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI condition: >50 for bullish, <50 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Stochastic filter: avoid overbought/oversold extremes
        stoch_not_overbought = stoch_d[i] < 80
        stoch_not_oversold = stoch_d[i] > 20
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA up, RSI bullish, not overbought, weekly uptrend, volatility filter
            if kama_up and rsi_bullish and stoch_not_overbought and weekly_uptrend and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI bearish, not oversold, weekly downtrend, volatility filter
            elif kama_down and rsi_bearish and stoch_not_oversold and weekly_downtrend and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA turns down or RSI < 50 or overbought or weekly trend fails
            if not (kama_up and rsi_bullish and stoch_not_overbought and weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up or RSI > 50 or oversold or weekly trend fails
            if not (kama_down and rsi_bearish and stoch_not_oversold and weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals