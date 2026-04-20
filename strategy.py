#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_VolumeATRFilter
# Hypothesis: 1h breakouts from daily Camarilla R1/S1 levels with volume confirmation and ATR filter
# work in both bull and bear markets by capturing institutional breakouts. Uses 1d for direction,
# 1h for entry timing to avoid excessive trading. Target: 15-37 trades/year.

name = "1h_Camarilla_R1_S1_Breakout_VolumeATRFilter"
timeframe = "1h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Pivot and Camarilla levels
    p = (ph + pl + pc) / 3
    r1 = p + (ph - pl) * 1.1 / 12
    s1 = p - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
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
    volatility_filter = atr > (atr_ma50 * 0.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Require session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + sufficient volatility
            if close[i] > r1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume confirmation + sufficient volatility
            elif close[i] < s1_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals