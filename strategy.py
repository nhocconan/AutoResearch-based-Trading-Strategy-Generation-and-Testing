#!/usr/bin/env python3
# 1d_KAMA_Trend_With_1W_Trend_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 1d chart adapts to market noise, reducing whipsaw in sideways markets while capturing trends. Combined with 1w EMA34 as a higher-timeframe trend filter to avoid counter-trend trades. Volume confirmation ensures institutional participation. Designed for low trade frequency (~10-25 trades/year) to minimize fee drag and improve generalization to bear markets.

name = "1d_KAMA_Trend_With_1W_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=1)).rolling(window=er_period, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate KAMA on 1d
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate 1d volume spike confirmation (volume > 1.5x 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34 and KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1w EMA34
            uptrend = close[i] > ema34_1w_aligned[i]
            downtrend = close[i] < ema34_1w_aligned[i]
            
            # Long: uptrend + price > KAMA + volume spike
            if uptrend and close[i] > kama[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price < KAMA + volume spike
            elif downtrend and close[i] < kama[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens or price crosses below KAMA
            if close[i] < ema34_1w_aligned[i] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens or price crosses above KAMA
            if close[i] > ema34_1w_aligned[i] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals