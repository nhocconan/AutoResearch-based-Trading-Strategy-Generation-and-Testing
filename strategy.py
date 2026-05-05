#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume spike and 1d EMA50 trend filter
# Long when price breaks above 4h Donchian upper (20) AND 1h volume > 2.0x 20-period average AND 1d EMA50 rising
# Short when price breaks below 4h Donchian lower (20) AND 1h volume > 2.0x 20-period average AND 1d EMA50 falling
# Exit when price returns to 4h Donchian midpoint OR 1d EMA50 flips direction
# Uses 4h/1d for signal direction, 1h only for entry timing precision. Target: 20-60 trades/year.
# Donchian provides structure, volume confirms participation, 1d EMA50 filters for higher-timeframe trend.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Session filter (08-20 UTC) reduces noise and overtrading.

name = "1h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) using completed 4h bars only
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper: highest high over past 20 completed 4h bars
    donchian_20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over past 20 completed 4h bars
    donchian_20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of upper and lower
    donchian_20_mid = (donchian_20_high + donchian_20_low) / 2
    
    # Align 4h Donchian levels to 1h timeframe (completed 4h bar only)
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_high)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_low)
    donchian_20_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_mid)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising when current EMA > previous EMA
    ema_50_rising = ema_50 > np.concatenate([[np.nan], ema_50[:-1]])
    # Falling when current EMA < previous EMA
    ema_50_falling = ema_50 < np.concatenate([[np.nan], ema_50[:-1]])
    
    # Align 1d EMA50 trend to 1h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(donchian_20_high_aligned[i]) or 
            np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(donchian_20_mid_aligned[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian upper AND volume spike AND 1d EMA50 rising
            if (close[i] > donchian_20_high_aligned[i] and 
                volume_filter[i] and 
                ema_50_rising_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below 4h Donchian lower AND volume spike AND 1d EMA50 falling
            elif (close[i] < donchian_20_low_aligned[i] and 
                  volume_filter[i] and 
                  ema_50_falling_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint OR 1d EMA50 starts falling
            if (close[i] < donchian_20_mid_aligned[i] or 
                ema_50_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint OR 1d EMA50 starts rising
            if (close[i] > donchian_20_mid_aligned[i] or 
                ema_50_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals