#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Spike
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with volume spike confirmation and 1w trend filter.
Long when price breaks above R1 with volume spike and 1w uptrend; short when breaks below S1 with volume spike and 1w downtrend.
Camarilla levels from prior 1d, volume spike >2.5x average, 1w EMA trend filter.
Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: 1w trend filter avoids counter-trend trades, high volume threshold filters false breakouts.
"""

name = "12h_Camarilla_R1S1_Breakout_Volume_Spike"
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
    
    # Get 1d data for Camarilla calculation (prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA40 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema40_1w = ema(close_1w, 40)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(n):
        # Get prior 1d OHLC (use previous 1d bar that's complete)
        # Since we're on 12h timeframe, we need to map to prior 1d bar
        # Use the 1d data aligned to 12h timeframe
        if i >= 2:  # Need at least 2 bars for prior day
            # Find the 1d bar that ended prior to current 12h bar
            # Simple approach: use 1d data shifted by 1 bar (prior completed day)
            pass  # Will calculate in vectorized form below
    
    # Vectorized Camarilla calculation using prior 1d bar
    # Shift 1d data by 1 to get prior completed day
    if len(df_1d) >= 2:
        prior_close = df_1d['close'].shift(1).values  # Prior day close
        prior_high = df_1d['high'].shift(1).values    # Prior day high
        prior_low = df_1d['low'].shift(1).values      # Prior day low
        
        # Calculate Camarilla levels for prior day
        camarilla_r1_1d = prior_close + (prior_high - prior_low) * 1.1 / 12
        camarilla_s1_1d = prior_close - (prior_high - prior_low) * 1.1 / 12
        
        # Align to 12h timeframe (each 12h bar = 0.5 of 1d bar)
        camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
        camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Calculate volume spike (volume > 2.5x 20-period average for strict confirmation)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 1w uptrend (close > EMA40)
            if close[i] > camarilla_r1[i] and volume_spike[i] and close[i] > ema40_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND 1w downtrend (close < EMA40)
            elif close[i] < camarilla_s1[i] and volume_spike[i] and close[i] < ema40_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 1w trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 1w trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals