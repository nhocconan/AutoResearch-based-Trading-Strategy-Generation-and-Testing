#!/usr/bin/env python3
"""
1h_adaptive_trend_pullback_v1
Hypothesis: Uses 4h EMA50 for trend direction, 1d EMA200 for regime filter, and RSI(14) for entry timing on 1h timeframe.
Enters long on RSI pullbacks to 40-50 in uptrends (price > 4h EMA50 and price > 1d EMA200) and short on RSI pullbacks to 50-60 in downtrends (price < 4h EMA50 and price < 1d EMA200).
Applies session filter (08-20 UTC) to reduce noise. Uses discrete position sizing (0.20) to minimize churn.
Designed to capture trend continuation moves while avoiding choppy markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adaptive_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after 1d EMA200 warmup
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 40 (end of pullback) or trend breaks
            if rsi[i] < 40 or close[i] < ema_50_4h_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 60 (end of pullback) or trend breaks
            if rsi[i] > 60 or close[i] > ema_50_4h_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI pullback to 40-50 in uptrend (price above both EMAs)
            if (40 <= rsi[i] <= 50 and 
                close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI pullback to 50-60 in downtrend (price below both EMAs)
            elif (50 <= rsi[i] <= 60 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals