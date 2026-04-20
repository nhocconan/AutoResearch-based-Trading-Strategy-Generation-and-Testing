# NOTE: This strategy will be overridden by the system prompt.
# The system prompt provides the exact code that must be used.
# The following is a placeholder and will be replaced.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (ATR, pivot levels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Daily Pivot levels (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.0)
    S1 = pivot_1d - (range_1d * 1.0)
    R2 = pivot_1d + (range_1d * 2.0)
    S2 = pivot_1d - (range_1d * 2.0)
    
    # Align 1d indicators to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Load 6h data for entry timing, volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume spike detection (20-period on 6h)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_14_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        atr = atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2, with volume confirmation and ATR filter
            if (price > R2_aligned[i] and 
                vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2, with volume confirmation and ATR filter
            elif (price < S2_aligned[i] and 
                  vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below R1 or ATR-based stop
            if price < R1_aligned[i] or price < (high_6h[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above S1 or ATR-based stop
            if price > S1_aligned[i] or price > (low_6h[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R2S2_Breakout_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0