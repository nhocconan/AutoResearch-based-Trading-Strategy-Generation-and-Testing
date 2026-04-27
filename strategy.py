#!/usr/bin/env python3
"""
#100984 - 1d_WickReversal_Volume_Spike_TrendFilter
Hypothesis: Capture mean-reversion at daily wicks with volume confirmation and weekly trend filter.
Long when price closes above previous day's low after testing it (bullish rejection), short when closes below previous day's high (bearish rejection).
Requires volume spike (>2x 20-period average) and alignment with weekly EMA20 trend.
Works in ranging markets (wick reversals) and trending markets (pullbacks to weekly trend).
Target: 10-20 trades/year to minimize fee drift. Uses discrete sizing (0.25).
"""

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
    
    # Get daily data for wick reversal signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align daily levels to 1d timeframe (already aligned, but for clarity)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Weekly trend filter: EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: bullish rejection of previous day's low (close > prev_low) with volume spike and above weekly EMA
        if (close[i] > prev_low_aligned[i] and
            volume_spike[i] and
            close[i] > ema20_1w_aligned[i]):
            signals[i] = 0.25
            position = 1
        # Short: bearish rejection of previous day's high (close < prev_high) with volume spike and below weekly EMA
        elif (close[i] < prev_high_aligned[i] and
              volume_spike[i] and
              close[i] < ema20_1w_aligned[i]):
            signals[i] = -0.25
            position = -1
        # Exit: reverse signal or price crosses weekly EMA (trend change)
        elif position == 1 and (close[i] < ema20_1w_aligned[i] or close[i] < prev_low_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > ema20_1w_aligned[i] or close[i] > prev_high_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WickReversal_Volume_Spike_TrendFilter"
timeframe = "1d"
leverage = 1.0