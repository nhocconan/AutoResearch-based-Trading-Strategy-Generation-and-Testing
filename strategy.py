#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week EMA trend filter and volume confirmation.
# Donchian(20) provides clear breakout levels; 1-week EMA filters trend direction.
# Volume confirmation ensures breakouts have participation.
# Target: 10-25 trades per year (40-100 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(20) for 1w trend filter
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Align 1w EMA to 1d timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian(20) on 1d timeframe
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 1w EMA + volume confirmation
            if (price > highest_high[i] and
                price > ema_trend and
                vol > 1.5 * avg_vol):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low + below 1w EMA + volume confirmation
            elif (price < lowest_low[i] and
                  price < ema_trend and
                  vol > 1.5 * avg_vol):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below Donchian low or below 1w EMA
            if (price < lowest_low[i] or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above Donchian high or above 1w EMA
            if (price > highest_high[i] or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_EMA_Volume"
timeframe = "1d"
leverage = 1.0