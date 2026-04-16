#!/usr/bin/env python3
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
    
    # === Weekly Pivot Points (for 1d trend bias) ===
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot from previous week
    pp_w = (df_1w['high'].shift(1) + df_1w['low'].shift(1) + df_1w['close'].shift(1)) / 3
    r1_w = pp_w + (df_1w['high'].shift(1) - df_1w['low'].shift(1)) * 1.1 / 6
    s1_w = pp_w - (df_1w['high'].shift(1) - df_1w['low'].shift(1)) * 1.1 / 6
    
    # Align weekly levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w.values)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w.values)
    
    # === 6h Donchian Channel (breakout) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Close on opposite breakout or failed continuation ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low (failed breakout)
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high (failed breakdown)
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Weekly bias: only long above weekly R1, short below weekly S1
            weekly_bias_long = price > r1_w_aligned[i]
            weekly_bias_short = price < s1_w_aligned[i]
            
            # LONG: Breakout above Donchian high with volume spike and weekly bullish bias
            if price > highest_high[i] and vol_spike and weekly_bias_long:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Breakdown below Donchian low with volume spike and weekly bearish bias
            elif price < lowest_low[i] and vol_spike and weekly_bias_short:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivotBias_VolumeSpike"
timeframe = "6h"
leverage = 1.0