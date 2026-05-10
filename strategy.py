#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
Combined with daily trend filter (EMA34) and volume spike confirmation, this strategy captures
breakouts with institutional interest. Designed for 4h timeframe to target 25-50 trades/year,
balancing edge and trade frequency. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "4h_Camarilla_R1_S1_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h (no extra delay needed as they're based on prior day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average for spike detection (20-day average)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and volume average (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current 4h volume > 2.0x average daily volume (scaled)
        # 1 day = 6 x 4h bars, so scale daily volume to 4h equivalent
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_spike = volume[i] > vol_4h_equiv * 2.0
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume spike
            if high[i] > camarilla_r1_aligned[i] and uptrend_1d and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume spike
            elif low[i] < camarilla_s1_aligned[i] and downtrend_1d and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or trend fails
            if low[i] < camarilla_s1_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 or trend fails
            if high[i] > camarilla_r1_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals