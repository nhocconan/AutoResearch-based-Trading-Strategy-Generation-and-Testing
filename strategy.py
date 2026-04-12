#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Volume_Squeeze
Hypothesis: Use 12h EMA trend filter with 4h Camarilla pivot levels and volume squeeze detection.
Long when price closes above Camarilla H3 in uptrend with volume contraction followed by expansion.
Short when price closes below Camarilla L3 in downtrend with volume contraction followed by expansion.
Volume squeeze defined as current volume < 50% of 20-period average, expansion as volume > 150% of average.
Targets breakouts from low volatility periods in trending markets, effective in both bull (continuation) and bear (reversal) phases.
Target: 50-120 total trades over 4 years (12-30/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # === VOLUME AVERAGE FOR SQUEEZE DETECTION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === CAMARILLA PIVOT LEVELS (DAILY) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from 12h EMA21
        close_12h_arr = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
        trend_up = close_12h_aligned[i] > ema21_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema21_12h_aligned[i]
        
        # Volume squeeze and expansion
        vol_squeeze = volume[i] < (vol_ma[i] * 0.5)
        vol_expansion = volume[i] > (vol_ma[i] * 1.5)
        
        # Entry conditions: price closes beyond Camarilla levels with trend and volume expansion
        long_signal = (close[i] > h3_aligned[i] and 
                      trend_up and 
                      vol_expansion and
                      (not vol_squeeze or i < 30 or volume[i-1] < vol_ma[i-1] * 0.5))
        
        short_signal = (close[i] < l3_aligned[i] and 
                       trend_down and 
                       vol_expansion and
                       (not vol_squeeze or i < 30 or volume[i-1] < vol_ma[i-1] * 0.5))
        
        # Exit conditions: opposite Camarilla level or trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_aligned[i] or not trend_up))
        exit_short = (position == -1 and 
                     (close[i] > h3_aligned[i] or not trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals