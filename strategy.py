#!/usr/bin/env python3
# 1H_Camarilla_R1S1_4H_Trend_Filter
# Hypothesis: Trade Camarilla pivot breakouts in the direction of 4h trend during active hours.
# Uses Camarilla R1/S1 levels on 1h for entry, 4h EMA50 for trend filter, and session filter (08-20 UTC) to reduce noise.
# Works in bull/bear by following 4h trend and using pivot levels for precise entries.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).

name = "1H_Camarilla_R1S1_4H_Trend_Filter"
timeframe = "1h"
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
    
    # Previous day's high, low, close for Camarilla calculation
    # We'll use daily data to calculate Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using the standard Camarilla formula based on previous day's range
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla R1 and S1 levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 1h (they are constant for the day)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: price above/below EMA50
    uptrend_4h = close_4h > ema50_4h
    downtrend_4h = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    uptrend_4h_1h = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_1h = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(uptrend_4h_1h[i]) or np.isnan(downtrend_4h_1h[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_level = r1_1h[i]
        s1_level = s1_1h[i]
        uptrend = uptrend_4h_1h[i] > 0.5
        downtrend = downtrend_4h_1h[i] > 0.5
        
        if position == 0:
            # Enter long: 4h uptrend + price breaks above R1
            if uptrend and close[i] > r1_level and close[i-1] <= r1_level:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + price breaks below S1
            elif downtrend and close[i] < s1_level and close[i-1] >= s1_level:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend weakens or price breaks below S1 (mean reversion)
            if not uptrend or close[i] < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend weakens or price breaks above R1 (mean reversion)
            if not downtrend or close[i] > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals