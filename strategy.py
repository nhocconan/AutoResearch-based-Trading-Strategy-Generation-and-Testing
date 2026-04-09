#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v11
# Hypothesis: 4-hour breakouts above/below daily Camarilla H4/L4 levels with volume confirmation and volatility filter.
# Uses tighter filters to reduce trade frequency and avoid fee drift. Only trades when volatility is moderate and volume is elevated.
# Exit when price returns to daily pivot (PP). Works in bull/bear markets as pivots adapt to volatility.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v11"
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
    
    # True Range and ATR(20)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # H4 and L4 levels (strong breakout levels)
    h4_1d = close_1d + (range_1d * 1.1 / 2)
    l4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume indicators
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Volume spike filter: current volume > 2.5x 20-period average
    vol_spike = volume > vol_ma_20 * 2.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR < 2% of price (avoid extreme volatility)
        vol_filter = atr[i] < 0.02 * close[i]
        
        # Volume confirmation: need volume spike OR above average volume
        vol_ok = volume[i] > vol_ma_20[i] * 1.5  # At least 1.5x average volume
        
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
            # Enter long: price breaks above H4 with volume and volatility filters
            if close[i] > h4_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 with volume and volatility filters
            elif close[i] < l4_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals