#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w Supertrend(ATR=10,mult=3) + volume confirmation
# Donchian breakouts capture momentum; 1w Supertrend shows weekly trend direction
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: Supertrend adapts to higher timeframe trend with ATR-based stops
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25-0.30

name = "1d_1w_donchian_supertrend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ATR(10) for Supertrend
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        if i < 10:
            atr[i] = np.nan
        elif i == 10:
            atr[i] = np.mean(tr[:11])
        else:
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Calculate Supertrend
    hl2 = (df_1w['high'].values + df_1w['low'].values) / 2
    upperband = hl2 + 3 * atr
    lowerband = hl2 - 3 * atr
    supertrend = np.full(len(close), np.nan)
    direction = np.full(len(close), np.nan)  # 1=up, -1=down
    
    for i in range(len(close)):
        if i < 10 or np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        if i == 10:
            supertrend[i] = upperband[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upperband[i-1]:
                supertrend[i] = upperband[i] if close[i] <= upperband[i] else lowerband[i]
                direction[i] = -1 if supertrend[i] == upperband[i] else 1
            else:
                supertrend[i] = lowerband[i] if close[i] >= lowerband[i] else upperband[i]
                direction[i] = 1 if supertrend[i] == lowerband[i] else -1
    
    # Align Supertrend direction to 1d timeframe (wait for weekly close)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(direction_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or direction_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or direction_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + weekly trend filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND weekly trend up
                if close[i] > donchian_high[i] and direction_aligned[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND weekly trend down
                elif close[i] < donchian_low[i] and direction_aligned[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals