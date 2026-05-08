#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA100 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper band (weekly) AND 1w EMA100 rising AND volume > 1.5x 50-period average.
# Short when price breaks below Donchian lower band (weekly) AND 1w EMA100 falling AND volume > 1.5x 50-period average.
# Exit when price crosses back inside the Donchian channel.
# Weekly Donchian provides strong institutional support/resistance. EMA100 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian_20_1wEMA100_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period high/low) from weekly data
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # 1w EMA100 for trend filter
    ema100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # 1w EMA100 direction
    ema100_rising = np.zeros_like(ema100_1w_aligned, dtype=bool)
    ema100_falling = np.zeros_like(ema100_1w_aligned, dtype=bool)
    ema100_rising[1:] = ema100_1w_aligned[1:] > ema100_1w_aligned[:-1]
    ema100_falling[1:] = ema100_1w_aligned[1:] < ema100_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 50-period average
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Sufficient warmup for EMA100 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema100_1w_aligned[i]) or np.isnan(ema100_rising[i]) or 
            np.isnan(ema100_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band, 1w EMA100 rising, volume filter
            long_cond = (close[i] > upper_band_aligned[i]) and ema100_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower band, 1w EMA100 falling, volume filter
            short_cond = (close[i] < lower_band_aligned[i]) and ema100_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower band
            if close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper band
            if close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals