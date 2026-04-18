#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_1dTrend_v2
Hypothesis: Camarilla pivot levels (R1, S1) from daily timeframe act as significant support/resistance. 
Price breaking above R1 with volume spike and above daily EMA34 indicates bullish momentum; 
breaking below S1 with volume spike and below daily EMA34 indicates bearish momentum. 
Daily EMA34 filter ensures alignment with medium-term trend, reducing false breakouts. 
Designed for low trade frequency (<30/year) to minimize fee drag while capturing meaningful moves.
"""

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
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: >2.0x 24-period average (2 periods = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 48  # Need EMA and volume MA
    
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
            # Long: price > R1 with volume spike and above daily EMA34
            if price > r1_val and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and below daily EMA34
            elif price < s1_val and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or below daily EMA34
            if price < s1_val or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or above daily EMA34
            if price > r1_val or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_1dTrend_v2"
timeframe = "12h"
leverage = 1.0