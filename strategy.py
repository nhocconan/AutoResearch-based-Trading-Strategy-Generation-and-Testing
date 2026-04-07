#!/usr/bin/env python3
"""
1d_ema_trend_reversion_v1
Hypothesis: On 1d timeframe, enter long when price crosses above EMA20 with RSI < 60 (mean reversion in uptrend) and enter short when price crosses below EMA20 with RSI > 40 (mean reversion in downtrend). Exit on opposite cross. Uses 1w EMA200 trend filter to avoid counter-trend trades. Designed for 10-30 trades/year to minimize fee flood while capturing mean reversion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_trend_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Calculate EMA20
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate RSI (14-period)
    if len(close) < 14:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(rsi[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20
            if close[i] < ema_20[i] and close[i-1] >= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20
            if close[i] > ema_20[i] and close[i-1] <= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price crosses above EMA20 with RSI < 60 and above weekly EMA200
            if close[i] > ema_20[i] and close[i-1] <= ema_20[i-1] and rsi[i] < 60 and close[i] > ema_200_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price crosses below EMA20 with RSI > 40 and below weekly EMA200
            elif close[i] < ema_20[i] and close[i-1] >= ema_20[i-1] and rsi[i] > 40 and close[i] < ema_200_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals