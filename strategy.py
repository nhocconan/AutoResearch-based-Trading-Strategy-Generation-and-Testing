#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_With_1dTrend
Hypothesis: On 12h timeframe, breakouts above/below Camarilla R1/S1 levels with volume confirmation and 1d EMA100 trend filter. This combines daily trend context with intraday breakout precision, reducing false signals in chop while capturing strong directional moves. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 from previous day
    rang = phigh - plow
    r1 = pclose + rang * 1.1 / 12
    s1 = pclose - rang * 1.1 / 12
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA100 trend filter
    ema_1d = pd.Series(pclose).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Need EMA100 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > R1 with volume spike and above 1d EMA100
            if price > r1_val and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below 1d EMA100
            elif price < s1_val and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or below 1d EMA100
            if price < s1_val or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or above 1d EMA100
            if price > r1_val or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_With_1dTrend"
timeframe = "12h"
leverage = 1.0