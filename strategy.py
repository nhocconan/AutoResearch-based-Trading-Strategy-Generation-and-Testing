#!/usr/bin/env python3
"""
4h_HTF_1d_Camarilla_R1S1_Breakout_VolumeFilter_V1
Hypothesis: Use 4h primary timeframe with 1d Camarilla R1/S1 breakout for high-probability momentum.
Add volume confirmation (>1.5x 30-bar volume MA) to avoid false breakouts.
Position size 0.25 balances risk/return. Target 20-50 trades/year per symbol.
Works in bull/bear via breakout logic and volume filter reducing whipsaw in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0  # R1 = Close + 1.1*(High-Low)/4
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0  # S1 = Close - 1.1*(High-Low)/4
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (30-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above 1d R1 + volume confirmation
            if price > r1_1d_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S1 + volume confirmation
            elif price < s1_1d_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below 1d pivot (mean reversion)
            pivot_val = pivot_1d[i//16] if i//16 < len(df_1d) else np.nan
            if not np.isnan(pivot_val) and price < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above 1d pivot (mean reversion)
            pivot_val = pivot_1d[i//16] if i//16 < len(df_1d) else np.nan
            if not np.isnan(pivot_val) and price > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_Camarilla_R1S1_Breakout_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0