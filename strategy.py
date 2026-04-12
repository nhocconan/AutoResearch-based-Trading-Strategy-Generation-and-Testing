#!/usr/bin/env python3
"""
1h_4d_1d_RSI_Trend_Filter
Hypothesis: On 1h timeframe, enter long when RSI(14) < 30 and 4h close > 1d EMA50, 
enter short when RSI(14) > 70 and 4h close < 1d EMA50. Uses 4h for trend direction 
and 1d EMA50 as stronger trend filter. RSI extremes provide mean-reversion entries 
within the trend. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). 
Target: 10-25 trades per year per symbol (40-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_RSI_Trend_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1H INDICATORS: RSI(14) ===
    # RSI with proper handling
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4H INDICATOR: Close for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # === 1D INDICATOR: EMA(50) for stronger trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(rsi[i]) or np.isnan(close_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters
        uptrend_4h = close_4h_aligned[i] > ema_50_1d_aligned[i]
        downtrend_4h = close_4h_aligned[i] < ema_50_1d_aligned[i]
        
        # Entry signals: RSI extremes in trend direction
        long_signal = uptrend_4h and rsi[i] < 30
        short_signal = downtrend_4h and rsi[i] > 70
        
        # Exit conditions: trend reversal or RSI normalization
        exit_long = not uptrend_4h or rsi[i] > 50
        exit_short = not downtrend_4h or rsi[i] < 50
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals