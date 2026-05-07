#!/usr/bin/env python3
"""
4H_Elder_Ray_Bull_Bear_2_1D_EMA34_Power
Hypothesis: Elder Ray Power (2-period EMA - 34-period EMA) on 1d determines bull/bear regime.
In bull regime (Power > 0), go long when 4h high touches 1d high and volume > 1.5x average.
In bear regime (Power < 0), go short when 4h low touches 1d low and volume > 1.5x average.
Volume confirmation reduces false breaks. Works in both bull (buy strength) and bear (sell weakness).
"""
name = "4H_Elder_Ray_Bull_Bear_2_1D_EMA34_Power"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA2 and EMA34
    close_1d = pd.Series(df_1d['close'])
    ema2 = close_1d.ewm(span=2, adjust=False).mean().values
    ema34 = close_1d.ewm(span=34, adjust=False).mean().values
    power = ema2 - ema34  # Elder Ray Power: bullish when >0, bearish when <0
    power_aligned = align_htf_to_ltf(prices, df_1d, power)
    
    # Get 1d high and low for reference levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = 34  # Ensure EMA34 is valid
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(power_aligned[i]) or np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 12 bars between trades (2 days on 4h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Bull regime: Power > 0, look for long
            if power_aligned[i] > 0:
                # Long: 4h high touches 1d high with volume confirmation
                if high[i] >= high_1d_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
            # Bear regime: Power < 0, look for short
            elif power_aligned[i] < 0:
                # Short: 4h low touches 1d low with volume confirmation
                if low[i] <= low_1d_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                    bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite 1d level or power flips
            if position == 1:
                if low[i] <= low_1d_aligned[i] or power_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
            elif position == -1:
                if high[i] >= high_1d_aligned[i] or power_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals