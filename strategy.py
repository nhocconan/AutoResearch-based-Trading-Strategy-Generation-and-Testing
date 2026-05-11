#!/usr/bin/env python3
name = "6h_ElderRay_ForceIndex_1dTrend"
timeframe = "6h"
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
    
    # Get 1d data for Elder Ray and Force Index (Elder's system)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 13-period EMA for Elder Ray calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power and Bear Power
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Force Index: (Close - Close_prev) * Volume
    close_diff_1d = np.diff(close_1d, prepend=close_1d[0])
    force_index_1d = close_diff_1d * volume_1d
    
    # Smooth Force Index with 2-period EMA for signal line
    force_ema2_1d = pd.Series(force_index_1d).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Get 6h data for entry timing
    close_6h = close
    high_6h = high
    low_6h = low
    volume_6h = volume
    
    # 6h RSI(14) for overbought/oversold filter
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_6h = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    force_ema2_aligned = align_htf_to_ltf(prices, df_1d, force_ema2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(force_ema2_aligned[i]) or
            np.isnan(rsi_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + Force Index rising + RSI not overbought
            if (bull_power_aligned[i] > 0 and 
                force_ema2_aligned[i] > force_ema2_aligned[i-1] and
                rsi_6h[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + Force Index falling + RSI not oversold
            elif (bear_power_aligned[i] < 0 and 
                  force_ema2_aligned[i] < force_ema2_aligned[i-1] and
                  rsi_6h[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Force Index turns down
            if (bull_power_aligned[i] <= 0 or 
                force_ema2_aligned[i] < force_ema2_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR Force Index turns up
            if (bear_power_aligned[i] >= 0 or 
                force_ema2_aligned[i] > force_ema2_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals