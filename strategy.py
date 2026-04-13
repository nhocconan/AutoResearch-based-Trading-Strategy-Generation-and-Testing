#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation.
# Uses weekly trend direction to filter breakouts in the direction of higher timeframe momentum.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 15-35 trades per year (60-140 total over 4 years) for 12h timeframe.
# Works in bull markets (breakouts with trend) and bear markets (avoids counter-trend breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume + above weekly EMA
            if (price > highest_high[i] and
                volume_confirm and
                price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower band with volume + below weekly EMA
            elif (price < lowest_low[i] and
                  volume_confirm and
                  price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or breaks below lower band
            mid = (highest_high[i] + lowest_low[i]) / 2
            if (price < mid or
                price < lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or breaks above upper band
            mid = (highest_high[i] + lowest_low[i]) / 2
            if (price > mid or
                price > highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_Trend_Filter_Volume_v1"
timeframe = "12h"
leverage = 1.0