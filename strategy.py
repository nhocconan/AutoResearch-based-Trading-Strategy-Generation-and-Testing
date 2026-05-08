#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d volume confirmation.
# Long when price breaks above 4h Donchian(20) high AND 1d volume > 1.2x 20-period average.
# Short when price breaks below 4h Donchian(20) low AND 1d volume > 1.2x 20-period average.
# Exit when price crosses back inside the 4h Donchian channel.
# Uses 4h for signal direction and 1d volume for confirmation, 1h only for entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency to avoid fee drag.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Donchian_20_1dVolume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20)
    donchian_period = 20
    upper_dc_4h = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc_4h = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    upper_dc = align_htf_to_ltf(prices, df_4h, upper_dc_4h)
    lower_dc = align_htf_to_ltf(prices, df_4h, lower_dc_4h)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.2 * vol_ma20_1d)
    volume_filter = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian upper, volume filter, session active
            long_cond = (close[i] > upper_dc[i]) and volume_filter[i]
            # Short conditions: price breaks below 4h Donchian lower, volume filter, session active
            short_cond = (close[i] < lower_dc[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 4h Donchian lower
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above 4h Donchian upper
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals