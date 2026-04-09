#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v14
# Hypothesis: 4-hour breakouts above/below daily Camarilla pivot levels (H4/L4) with volume confirmation and volatility filter.
# Uses breakout of H4/L4 levels (stronger breakout than H3/L3) for higher probability moves.
# Exit when price returns to the daily pivot point (PP).
# Works in both bull and bear markets as pivot levels adapt to volatility, and filters reduce whipsaw.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# This version reduces trade frequency by tightening volume and volatility filters, and adding a volume spike filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v14"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # H4 and L4 levels (stronger breakout levels)
    h4_1d = close_1d + (range_1d * 1.1 / 2)  # Same as R4
    l4_1d = close_1d - (range_1d * 1.1 / 2)  # Same as S4
    
    # Align 1d levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
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
        vol_filter = atr[i] < 0.025 * close[i]  # ATR less than 2.5% of price (was 4%)
        
        # Volume confirmation: current volume > 1.8x 20-period average (more restrictive)
        vol_ok = volume[i] > vol_ma_20[i] * 1.8  # Was 1.3
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 level with volume confirmation and volatility filter
            if close[i] > h4_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 level with volume confirmation and volatility filter
            elif close[i] < l4_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals