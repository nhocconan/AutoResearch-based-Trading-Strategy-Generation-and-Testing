#!/usr/bin/env python3
"""
12h_Weekly_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: 12h price breaks above/below weekly Camarilla pivot levels (R1/S1) with volume confirmation and daily EMA34 trend filter.
Designed to capture high-probability breakouts in both bull and bear markets by combining weekly structure, volume, and trend alignment.
Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drift and maximize edge.
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
    volume = prices['volume'].values
    
    # --- Weekly HTF Camarilla Pivots (from weekly high/low/close) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using weekly OHLC
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    camarilla_width = (wk_high - wk_low) * 1.1 / 12
    r1 = wk_close + camarilla_width
    s1 = wk_close - camarilla_width
    
    # Align to 12h timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # --- Daily HTF EMA34 Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- 12h Indicators ---
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume in uptrend (price > EMA34)
            if price > r1_level and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume in downtrend (price < EMA34)
            elif price < s1_level and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to midpoint of R1-S1 or trend reverses
            midpoint = (r1_level + s1_level) / 2
            if price < midpoint or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to midpoint of R1-S1 or trend reverses
            midpoint = (r1_level + s1_level) / 2
            if price > midpoint or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Weekly_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0