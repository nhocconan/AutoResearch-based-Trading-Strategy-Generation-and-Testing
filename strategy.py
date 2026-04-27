#!/usr/bin/env python3
"""
6h_Aroon_Oscillator_12hTrend_Filter
Hypothesis: Aroon oscillator (down-up) identifies trend strength on 6h. 
Filter with 12h EMA50 trend to avoid counter-trend trades. 
Enter when Aroon > 50 (strong uptrend) or < -50 (strong downtrend) 
with price pulling back to 20 EMA on 6h. Works in bull via pullbacks in uptrend,
in bear via bounces in downtrend. Target: 20-35 trades/year.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Aroon oscillator (25-period) on 6h
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        high_idx = np.argmax(window_high)
        low_idx = np.argmin(window_low)
        aroon_up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # 20 EMA on 6h for pullback entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Aroon and EMAs
    start_idx = max(period - 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_osc[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        aroon_val = aroon_osc[i]
        ema_20_val = ema_20[i]
        ema_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long: strong uptrend (Aroon > 50) + pullback to EMA20 + 12h uptrend
            if aroon_val > 50 and close_val <= ema_20_val * 1.001 and ema_12h_val < close_val:
                signals[i] = size
                position = 1
            # Short: strong downtrend (Aroon < -50) + bounce to EMA20 + 12h downtrend
            elif aroon_val < -50 and close_val >= ema_20_val * 0.999 and ema_12h_val > close_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: trend weakness (Aroon < 0) or 12h trend change
            if aroon_val < 0 or ema_12h_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend weakness (Aroon > 0) or 12h trend change
            if aroon_val > 0 or ema_12h_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Aroon_Oscillator_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0