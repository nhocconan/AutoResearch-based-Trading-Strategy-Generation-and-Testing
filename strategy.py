#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_VolumeATR_Filter_V1
Breakout of Camarilla R1/S1 levels from prior 1d candle with volume surge and ATR-based volatility filter.
Long when price breaks above R1 with volume > 1.5x 20-period average and ATR < 0.8x 20-period average.
Short when price breaks below S1 with same filters.
Exit on close crossing back below R1 (long) or above S1 (short).
Position size: 0.20. Target: 15-37 trades/year.
Uses 1d for signal direction, 1h for entry timing. Session filter (08-20 UTC) to reduce noise.
Works in bull/bear: breakouts capture momentum, volume filter avoids false breakouts, ATR filter avoids high volatility chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rang = high_1d - low_1d
    r1_1d = close_1d + 1.1 * rang / 12
    s1_1d = close_1d - 1.1 * rang / 12
    
    # Align to 1h timeframe (previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: 20-period average on 1d
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # ATR filter: 14-period ATR on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(40, n):  # warmup for rolling averages
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_ma20_1d_aligned[i]) or np.isnan(atr_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d values aligned to 1h
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        atr_1d_current = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        atr_filter = atr_1d_current < (0.8 * atr_ma20_1d_aligned[i])  # low volatility
        
        if position == 0:
            # Long breakout above R1
            if close[i] > r1_1d_aligned[i] and volume_filter and atr_filter:
                signals[i] = 0.20
                position = 1
            # Short breakdown below S1
            elif close[i] < s1_1d_aligned[i] and volume_filter and atr_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close back below R1
            if close[i] < r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close back above S1
            if close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_VolumeATR_Filter_V1"
timeframe = "1h"
leverage = 1.0