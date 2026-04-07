#!/usr/bin/env python3
"""
1D Daily Donchian Breakout with Weekly Trend Filter
Long when price breaks above 20-day Donchian upper band AND weekly EMA trend up
Short when price breaks below 20-day Donchian lower band AND weekly EMA trend down
Exit when price crosses back to middle line (10-day EMA)
Target: 20-60 trades per year (~80-240 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle: 10-period EMA for exit
    ema_mid = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Weekly trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle line (10-day EMA)
            if close[i] < ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle line (10-day EMA)
            if close[i] > ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry: Donchian breakout with weekly trend filter
            if close[i] > donch_upper[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                # Breakout above upper band with rising weekly EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_lower[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                # Breakdown below lower band with falling weekly EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals