#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(50) trend filter and 1-day volume confirmation.
# Long when price breaks above Donchian upper band, price > EMA50 (12h), and 1d volume > 1.4x 20-period average.
# Short when price breaks below Donchian lower band, price < EMA50 (12h), and 1d volume > 1.4x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# Uses 4h timeframe with volume confirmation from higher timeframe to reduce false breakouts.
# Target: 80-160 total trades over 4 years (20-40/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_12hEMA50_1dVolume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volume filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 25:
        return np.zeros(n)
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian(20) on 4h data
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # EMA50 on 12h close
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Daily volume filter: current volume > 1.4x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.4 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, price > EMA50 (12h), volume confirmation
            long_cond = (close[i] > upper_dc[i]) and (close[i] > ema_12h_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, price < EMA50 (12h), volume confirmation
            short_cond = (close[i] < lower_dc[i]) and (close[i] < ema_12h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals