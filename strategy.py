#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h trend filter.
# Donchian channels provide clear breakout signals. Volume confirms institutional interest.
# 12h trend filter ensures we trade with the higher timeframe direction.
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.
# Works in bull markets (breakouts continue) and bear markets (failed breakouts reverse quickly).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) for breakouts
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (21 + 1)
    ema_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * ema_multiplier + ema_12h[i-1]
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(25, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        trend_12h = ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + above 12h EMA
            if (price > highest_high[i] and
                volume_confirm and
                price > trend_12h):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + below 12h EMA
            elif (price < lowest_low[i] and
                  volume_confirm and
                  price < trend_12h):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or breaks below low
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (price < donchian_mid or
                price < lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or breaks above high
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if (price > donchian_mid or
                price > highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0