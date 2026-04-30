#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.5x average)
# Donchian breakout captures momentum, 12h EMA50 filters for higher timeframe trend,
# volume confirmation reduces false breakouts. Works in bull/bear regimes via trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_Donchian20_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup for Donchian (20), EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_prev_close = close[i-1]
        curr_prev_high = high[i-1]
        curr_prev_low = low[i-1]
        curr_volume_spike = volume_spike[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Donchian breakout
            if curr_volume_spike:
                # Bullish: price breaks above Donchian upper + price above 12h EMA50
                if curr_high > highest_high[i-1] and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower + price below 12h EMA50
                elif curr_low < lowest_low[i-1] and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower (trend reversal)
            if curr_low < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper (trend reversal)
            if curr_high > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals