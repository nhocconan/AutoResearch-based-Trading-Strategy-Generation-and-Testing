#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Breakout of Camarilla R1/S1 levels with 12h trend filter and volume confirmation.
# Long when: price breaks above R1 in 12h uptrend with volume > 1.5x average.
# Short when: price breaks below S1 in 12h downtrend with volume > 1.5x average.
# Uses Camarilla levels from daily data for structure and 12h EMA50 for trend.
# Works in bull/bear by following 12h trend and using volume to confirm institutional interest.
# Target: 20-30 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily Camarilla levels (S1, R1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: based on previous day
    range_1d = high_1d - low_1d
    R1 = close_1d + range_1d * 1.1 / 12
    S1 = close_1d - range_1d * 1.1 / 12
    
    # Align daily levels to 4h (they are fixed for the day)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Trend direction from 12h EMA50
        uptrend_12h = close_12h >= ema50_12h  # Use actual 12h close for trend
        # We need to get the trend value at the corresponding 12h bar
        # Since we can't easily map i to 12h index here, we'll use the aligned EMA
        # Price above EMA = uptrend, below = downtrend
        uptrend_12h_aligned = close[i] > ema50_12h_aligned[i]
        downtrend_12h_aligned = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R1, 12h uptrend, volume confirmation
            if close[i] > R1_4h[i] and uptrend_12h_aligned and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, 12h downtrend, volume confirmation
            elif close[i] < S1_4h[i] and downtrend_12h_aligned and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or trend changes
            if close[i] < S1_4h[i] or not uptrend_12h_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or trend changes
            if close[i] > R1_4h[i] or not downtrend_12h_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals