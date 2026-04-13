#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Uses Donchian channels (20-period high/low) for breakout signals.
# Filters with 1-day EMA trend to avoid counter-trend trades.
# Requires volume > 1.5x average volume for confirmation.
# Designed to work in both bull and bear markets by using 1d trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(20) for 1d trend filter
    ema20_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (20 + 1)
    ema20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema20_1d[i] = (close_1d[i] - ema20_1d[i-1]) * ema_multiplier + ema20_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1d EMA20 + volume confirmation
            if (price > donch_high[i] and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + below 1d EMA20 + volume confirmation
            elif (price < donch_low[i] and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below 1d EMA
            if (price < donch_low[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above 1d EMA
            if (price > donch_high[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Trend_Volume"
timeframe = "4h"
leverage = 1.0