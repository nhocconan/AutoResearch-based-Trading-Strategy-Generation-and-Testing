# 4H_Supertrend_RSI_Range
# Hypothesis: Supertrend identifies trend direction while RSI identifies overbought/oversold conditions for mean reversion entries within the trend. Combines trend-following with mean-reversion to work in both bull and bear markets. Designed for low trade frequency with discrete sizing (0.25) to minimize fee drag.

name = "4H_Supertrend_RSI_Range"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend (ATR period 10, multiplier 3.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            trend[i] = 1
        elif close[i] < supertrend[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get 1d close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold (<30) and Supertrend uptrend and 1d uptrend
            if rsi[i] < 30 and trend[i] == 1 and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) and Supertrend downtrend and 1d downtrend
            elif rsi[i] > 70 and trend[i] == -1 and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (>70) or Supertrend turns down
            if rsi[i] > 70 or trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold (<30) or Supertrend turns up
            if rsi[i] < 30 or trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals