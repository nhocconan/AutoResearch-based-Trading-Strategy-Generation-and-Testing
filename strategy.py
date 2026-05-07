#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 12h EMA200 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) lower band AND 12h EMA200 falling AND volume > 1.5x 20-period average.
# Exit when price crosses the Donchian midpoint (average of upper and lower bands).
# This strategy captures volatility expansion in trending markets while filtering out chop.
# The 12h EMA200 ensures alignment with the long-term trend, reducing false breakouts.
# Volume confirmation ensures institutional participation.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend direction.

name = "4h_DonchianBreakout_12hEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channels (20)
    dc_length = 20
    highest_high = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    lowest_low = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    upper_band = highest_high
    lower_band = lowest_low
    midpoint = (upper_band + lower_band) / 2.0
    
    # 12h EMA200 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # 12h EMA200 direction
    ema200_rising = np.zeros_like(ema200_12h_aligned, dtype=bool)
    ema200_falling = np.zeros_like(ema200_12h_aligned, dtype=bool)
    ema200_rising[1:] = ema200_12h_aligned[1:] > ema200_12h_aligned[:-1]
    ema200_falling[1:] = ema200_12h_aligned[1:] < ema200_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 200)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(ema200_rising[i]) or np.isnan(ema200_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band, 12h EMA200 rising, volume filter
            long_cond = (close[i] > upper_band[i]) and ema200_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower band, 12h EMA200 falling, volume filter
            short_cond = (close[i] < lower_band[i]) and ema200_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below midpoint
            if close[i] < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above midpoint
            if close[i] > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals