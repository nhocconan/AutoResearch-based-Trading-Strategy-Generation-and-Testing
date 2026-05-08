#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above upper band AND 1w EMA34 rising AND volume > 2x 20-period average.
# Short when price breaks below lower band AND 1w EMA34 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside the Donchian channel (middle line).
# This strategy captures long-term breakouts with trend alignment and volume confirmation.
# Donchian channels provide clear breakout levels. The 1w EMA34 filter ensures we trade with the weekly trend.
# Volume spike confirms institutional participation.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1w trend direction.

name = "1d_Donchian_20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_max_20
    lower_band = low_min_20
    middle_band = (upper_band + lower_band) / 2.0
    
    # Calculate EMA34 on weekly data for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w EMA34 direction
    ema34_rising = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1w_aligned[1:] > ema34_1w_aligned[:-1]
    ema34_falling[1:] = ema34_1w_aligned[1:] < ema34_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band, 1w EMA34 rising, volume filter
            long_cond = (close[i] > upper_band[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower band, 1w EMA34 falling, volume filter
            short_cond = (close[i] < lower_band[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below middle band
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above middle band
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals