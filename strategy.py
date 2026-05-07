#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_BullTrend_1dVolume"
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
    
    # Get 1d data for Elder Ray Bull Power (13-period EMA) and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Volume filter: current 1d volume > 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_1d > vol_avg
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # Get 1d data for 13-period EMA trend filter
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or 
            np.isnan(ema13_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum), price above EMA13 (uptrend), volume confirmation
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema13_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum), price below EMA13 (downtrend), volume confirmation
            # Bear Power = Low - EMA13
            else:
                # Calculate bear power for short condition
                low_1d = df_1d['low'].values
                ema13_1d = ema13  # already calculated
                bear_power = low_1d - ema13_1d
                bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
                if not np.isnan(bear_power_aligned[i]) and \
                   (bear_power_aligned[i] < 0 and 
                    close[i] < ema13_aligned[i] and 
                    volume_filter_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (momentum shift)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (momentum shift)
            low_1d = df_1d['low'].values
            ema13_1d = ema13  # already calculated
            bear_power = low_1d - ema13_1d
            bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
            if not np.isnan(bear_power_aligned[i]) and bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals