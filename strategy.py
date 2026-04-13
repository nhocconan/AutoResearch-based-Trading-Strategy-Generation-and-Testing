#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and 1d EMA filter.
# Donchian channel breakouts capture momentum in trending markets.
# Volume confirmation filters out false breakouts.
# Daily EMA filter ensures trades align with higher timeframe trend.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate daily EMA(21) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (21 + 1)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * ema_multiplier + ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(25, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        daily_ema = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.4x average volume
        volume_confirm = vol > 1.4 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + above daily EMA
            if (price > highest_high[i] and 
                volume_confirm and
                price > daily_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + below daily EMA
            elif (price < lowest_low[i] and
                  volume_confirm and
                  price < daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches Donchian low (trailing stop)
            if price < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches Donchian high (trailing stop)
            if price > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0