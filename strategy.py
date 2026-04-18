#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Camarilla R1/S1 levels (from prior 1d) with volume confirmation and EMA34 trend filter. Uses 1d OHLC to calculate pivot levels, which are leading support/resistance. Volume > 1.5x 20-period average confirms breakout strength. EMA34 ensures trend alignment. Designed to work in both bull (breakouts above R1) and bear (breakdowns below S1) markets with tight entry conditions to avoid overtrading.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1day OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla R1 and S1
    r1_1d = c_1d + (h_1d - l_1d) * 1.1 / 12
    s1_1d = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align to 4h timeframe (use prior completed 1day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # EMA34 trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_34[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        vol_ok = volume_filter[i]
        ema34 = ema_34[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to midpoint of R1-S1 or trend reverses
            midpoint = (r1 + s1) / 2
            if price < midpoint or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to midpoint of R1-S1 or trend reverses
            midpoint = (r1 + s1) / 2
            if price > midpoint or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0