#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot levels (R1/S1) with 1d volume spike filter.
Long when price breaks above R1 with volume spike, short when breaks below S1 with volume spike.
Breakouts require institutional interest (volume spike) to avoid false signals.
Uses 1h timeframe for entry timing but 4h for pivot levels and 1d for volume filter.
Works in both bull and bear markets by following volume-confirmed breakouts.
Target: 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for Camarilla pivots - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas
    R1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    S1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align to 1h timeframe (wait for 4h bar to close)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Load 1d data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike (>1.5x average)
            if (prices['close'].iloc[i] > R1_4h_aligned[i] and 
                volume_1d[i] > vol_ma_1d_aligned[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume spike (>1.5x average)
            elif (prices['close'].iloc[i] < S1_4h_aligned[i] and 
                  volume_1d[i] > vol_ma_1d_aligned[i] * 1.5):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price returns to 4h midpoint
            midpoint_4h = (R1_4h_aligned[i] + S1_4h_aligned[i]) / 2
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls back to midpoint
                if prices['close'].iloc[i] < midpoint_4h:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises back to midpoint
                if prices['close'].iloc[i] > midpoint_4h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "4H_Camarilla_R1S1_VolumeSpike_1h"
timeframe = "1h"
leverage = 1.0