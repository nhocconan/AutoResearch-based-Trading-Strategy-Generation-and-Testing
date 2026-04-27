#!/usr/bin/env python3
"""
4h_SR_Reversal_1dTrend_Volume
Hypothesis: Combines dynamic support/resistance from 14-period swing highs/lows with 1d trend (EMA50) and volume spike (>2x 20-period average) to capture reversals at key levels. Works in both bull and bear markets by trading pullbacks to support/resistance in the direction of higher timeframe trend. Designed for low trade frequency (~20-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 14-period swing highs and lows for dynamic S/R
    window = 14
    highest_high = pd.Series(high).rolling(window=window, center=False).max().values
    lowest_low = pd.Series(low).rolling(window=window, center=False).min().values
    
    # 1d EMA50 for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for swing calculations and EMA
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        hh = highest_high[i]
        ll = lowest_low[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price pulls back to support (lowest_low) in uptrend with volume
            if close[i] >= ll * 0.998 and close[i] <= ll * 1.002 and close[i] > ema50 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price pulls back to resistance (highest_high) in downtrend with volume
            elif close[i] >= hh * 0.998 and close[i] <= hh * 1.002 and close[i] < ema50 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below 1d EMA50 or reaches resistance
            if close[i] < ema50 or close[i] >= hh * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above 1d EMA50 or reaches support
            if close[i] > ema50 or close[i] <= ll * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_SR_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0