#!/usr/bin/env python3
# 12h_WeeklyPivot_R2_S2_Breakout_VolumeATRFilter
# Hypothesis: Weekly pivot levels (R2/S2) act as strong institutional support/resistance.
# Breakouts from these levels with volume confirmation and ATR volatility filter capture
# sustained moves with fewer trades. Works in bull/bear markets by capturing breakouts
# from key weekly levels with institutional volume validation. Target: 50-150 total trades over 4 years.

name = "12h_WeeklyPivot_R2_S2_Breakout_VolumeATRFilter"
timeframe = "12h"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot and R2/S2 levels from previous week
    ph = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    
    # Weekly pivot and R2/S2
    p = (ph + pl + pc) / 3
    r2 = p + 2 * (ph - pl)
    s2 = p - 2 * (ph - pl)
    
    # Align weekly levels to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    # ATR filter: only trade when volatility is sufficient
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma50 * 0.5)  # Only trade when ATR > 50% of its MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 + volume confirmation + sufficient volatility
            if close[i] > r2_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 + volume confirmation + sufficient volatility
            elif close[i] < s2_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S2 (reversal signal)
            if close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R2 (reversal signal)
            if close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals