#!/usr/bin/env python3
"""
12h_rsi_ema_pullback_1d_trend_v1
Hypothesis: On 12h timeframe, buy pullbacks to EMA20 during uptrends (EMA50 > EMA200) and sell rallies to EMA20 during downtrends (EMA50 < EMA200), with RSI(14) confirming momentum (RSI < 40 for longs, RSI > 60 for shorts). Daily trend filter ensures alignment with higher timeframe trend. Target: 50-150 trades over 4 years (12-37/year) to balance opportunity with fee control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_ema_pullback_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # EMA calculations
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    daily_uptrend = ema50_1d > ema200_1d
    daily_downtrend = ema50_1d < ema200_1d
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(rsi[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or EMA50 < EMA20 (trend weakness)
            if rsi[i] > 60 or ema50[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or EMA50 > EMA20 (trend weakness)
            if rsi[i] < 40 or ema50[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: daily uptrend + price near EMA20 + RSI oversold
            if daily_uptrend_aligned[i] and close[i] <= ema20[i] * 1.005 and rsi[i] < 40:
                position = 1
                signals[i] = 0.25
            # Short: daily downtrend + price near EMA20 + RSI overbought
            elif daily_downtrend_aligned[i] and close[i] >= ema20[i] * 0.995 and rsi[i] > 60:
                position = -1
                signals[i] = -0.25
    
    return signals