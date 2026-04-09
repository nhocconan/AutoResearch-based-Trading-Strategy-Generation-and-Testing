#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Elder Ray (Bull/Bear Power) + volume confirmation
# Donchian breakouts capture momentum; 1d Elder Ray shows institutional buying/selling pressure
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: Elder Ray adapts to higher timeframe power balance
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_elder_ray_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (standard)
    close_1d = df_1d['close'].values
    ema13 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        multiplier = 2 / (13 + 1)
        ema13[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13[i] = (close_1d[i] * multiplier) + (ema13[i-1] * (1 - multiplier))
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Align Elder Ray data to 6h timeframe (wait for daily close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h Donchian channels (20-period)
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
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR Bear Power > 0 (bulls losing control)
            if close[i] < donchian_low[i] or bear_power_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR Bull Power < 0 (bears losing control)
            if close[i] > donchian_high[i] or bull_power_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Elder Ray filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND Bull Power > 0 (bulls in control)
                if close[i] > donchian_high[i] and bull_power_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND Bear Power < 0 (bears in control)
                elif close[i] < donchian_low[i] and bear_power_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals