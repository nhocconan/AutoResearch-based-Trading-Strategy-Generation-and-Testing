#!/usr/bin/env python3
# 1h_4d_camarilla_breakout_v1
# Hypothesis: 1-hour breakouts above/below 4-hour Camarilla pivot levels (H4/L4) with volume confirmation and volatility filter.
# Uses 4h Camarilla levels for directional bias (updated only when 4h candle closes) and 1h for precise entry timing.
# Exit when price returns to the 4h pivot point (PP).
# Works in both bull and bear markets as pivot levels adapt to volatility, and filters reduce whipsaw.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Position size: 0.20 (20% of capital).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Load 4h data ONCE before loop for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # H4 and L4 levels (stronger breakout levels)
    h4_4h = close_4h + (range_4h * 1.1 / 2)  # Same as R4
    l4_4h = close_4h - (range_4h * 1.1 / 2)  # Same as S4
    
    # Align 4h levels to 1h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_spike = volume > vol_ma_20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility (more restrictive)
        vol_filter = atr[i] < 0.025 * close[i]  # ATR less than 2.5% of price
        
        # Volume confirmation: current volume > 1.8x 20-period average (more restrictive)
        vol_ok = volume[i] > vol_ma_20[i] * 1.8
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above H4 level with volume confirmation and volatility filter
            if close[i] > h4_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L4 level with volume confirmation and volatility filter
            elif close[i] < l4_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.20
    
    return signals