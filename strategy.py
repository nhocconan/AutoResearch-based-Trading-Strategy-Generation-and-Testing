#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Reversal_1dTrend_VolumeSpike
Hypothesis: Trade reversals at extreme Camarilla levels (R4/S4) with 1d trend filter and volume confirmation.
Enters long when price closes below S4 (oversold) with volume confirmation and bullish 1d trend.
Enters short when price closes above R4 (overbought) with volume confirmation and bearish 1d trend.
Exits on midpoint (C) level retest. Uses discrete sizing (0.25) to minimize fee churn.
Targets 50-150 trades over 4 years by requiring confluence of extreme level, volume, and trend.
Works in ranging markets where price reverses at extremes, and in trends when pullbacks occur.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1d data for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter (smoother than EMA34)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels for 1d
    # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    # Midpoint (C) = close
    camarilla_r4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_c = close_1d  # Pivot point / midpoint
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period volume median, 50-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_c_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close below S4 (oversold) + volume confirm + bullish 1d trend
        if close[i] < camarilla_s4_aligned[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close above R4 (overbought) + volume confirm + bearish 1d trend
        elif close[i] > camarilla_r4_aligned[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price retests midpoint (C), short exits when price retests midpoint (C)
        elif position == 1 and close[i] >= camarilla_c_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] <= camarilla_c_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R4_S4_Reversal_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0