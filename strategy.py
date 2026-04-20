#!/usr/bin/env python3
# 6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: Daily pivot R1/S1 breakouts on 6h timeframe with volume confirmation and ATR filter.
# Uses daily pivots for key support/resistance, volume spike to confirm breakout strength,
# and ATR to avoid low-volatility false breakouts. Works in bull (breakouts continue) and bear (fails at resistance/support).
# Target: 15-30 trades/year per symbol.

name = "6h_1d_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate ATR(14) for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.absolute(low[1:] - close[:-1])
    tr = np.maximum(tr1, tr2)
    atr = np.concatenate([[np.nan], pd.Series(tr).rolling(window=14, min_periods=14).mean().values])
    
    # Calculate volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6 days
    
    # Align 1d indicators to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 1d average volume
        volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # ATR filter: avoid low volatility environments
        atr_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price breaks above R1 with volume and sufficient volatility
            if close[i] > r1_1d_aligned[i] and volume_spike and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and sufficient volatility
            elif close[i] < s1_1d_aligned[i] and volume_spike and atr_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals