#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA200 trend filter and volume confirmation.
# Long when price breaks above 20-period high AND EMA200(12h) rising AND volume > 1.5x 20-period avg.
# Short when price breaks below 20-period low AND EMA200(12h) falling AND volume > 1.5x 20-period avg.
# Exit on opposite breakout to avoid whipsaw in ranging markets.
# Uses structure-based breakouts with trend filter to capture trends while avoiding counter-trend trades.
# Target: 80-160 total trades over 4 years (20-40/year) for low fee drift.

name = "6h_Donchian20_12hEMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_rising = ema200_12h > np.roll(ema200_12h, 1)  # rising if current > previous
    ema200_12h_falling = ema200_12h < np.roll(ema200_12h, 1)  # falling if current < previous
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    ema200_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h_rising)
    ema200_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema200_12h_aligned[i]) or
            np.isnan(ema200_12h_rising_aligned[i]) or np.isnan(ema200_12h_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above 20-period high, EMA200 rising, volume spike
            long_cond = (close[i] > highest_high[i]) and ema200_12h_rising_aligned[i] and volume_filter[i]
            # Short conditions: breakdown below 20-period low, EMA200 falling, volume spike
            short_cond = (close[i] < lowest_low[i]) and ema200_12h_falling_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below 20-period low (reverse signal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above 20-period high (reverse signal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals