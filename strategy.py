#!/usr/bin/env python3
# 6h_1d_RSI_Trend_Bounce
# Hypothesis: On 6h timeframe, buy when price touches 1d EMA50 in uptrend with RSI(14) < 40,
# sell when price touches 1d EMA50 in downtrend with RSI(14) > 60. Uses daily trend as filter
# and 6s RSI for oversold/overbought bounces. Works in both bull (buy dips) and bear (sell rallies).
# Target: 15-30 trades/year.

name = "6h_1d_RSI_Trend_Bounce"
timeframe = "6h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure RSI and EMA50 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near 1d EMA50 (support) in uptrend with oversold RSI
            if close[i] > ema50_1d_aligned[i] and abs(close[i] - ema50_1d_aligned[i]) / ema50_1d_aligned[i] < 0.005 and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: price near 1d EMA50 (resistance) in downtrend with overbought RSI
            elif close[i] < ema50_1d_aligned[i] and abs(close[i] - ema50_1d_aligned[i]) / ema50_1d_aligned[i] < 0.005 and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI becomes overbought or price moves significantly above EMA50
            if rsi[i] > 70 or close[i] > ema50_1d_aligned[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI becomes oversold or price moves significantly below EMA50
            if rsi[i] < 30 or close[i] < ema50_1d_aligned[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals