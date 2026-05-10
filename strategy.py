#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike_v1
# Hypothesis: Price breaking above Camarilla R1 in a 1d uptrend (price above EMA50) with volume surge
# indicates institutional buying and continuation. Similarly, breaking below S1 in a 1d downtrend
# with volume surge indicates institutional selling. This works in both bull and bear markets by
# aligning with the higher timeframe trend, reducing false breakouts. Volume spike confirms
# institutional participation, filtering out low-conviction moves. Target: 20-40 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.12 / 12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.12 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: 1d uptrend + price breaks above R1 + volume spike
            if uptrend and close[i] > camarilla_r1_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: 1d downtrend + price breaks below S1 + volume spike
            elif downtrend and close[i] < camarilla_s1_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals