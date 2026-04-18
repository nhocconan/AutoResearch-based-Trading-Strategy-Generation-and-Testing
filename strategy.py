#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_With_Volume_And_Trend_Filter_v1
Hypothesis: Use daily Camarilla pivot levels (R1, S1) for mean-reversion entries on 4h, confirmed by volume spikes and aligned with 1-week EMA trend to work in both bull and bear markets. Camarilla levels provide high-probability reversal zones. Volume confirms institutional interest. Weekly EMA filter ensures trades align with higher timeframe trend, avoiding counter-trend whipsaws. Designed for low trade frequency (<30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1-week EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >2.0x 20-period average (high threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Price touches S1 with volume spike and above weekly EMA
            if price <= s1 and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: Price touches R1 with volume spike and below weekly EMA
            elif price >= r1 and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Price reaches midpoint (C) or touches R1
            midpoint = (r1 + s1) / 2
            if price >= midpoint or price >= r1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Price reaches midpoint (C) or touches S1
            midpoint = (r1 + s1) / 2
            if price <= midpoint or price <= s1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_With_Volume_And_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0