#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d/1w trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and rising + 1d trend up + volume > 1.5x avg
# Short when Bear Power > 0 and rising + 1d trend down + volume > 1.5x avg
# Exit when power turns negative or trend reverses
# Designed for 50-150 total trades over 4 years on 6h timeframe

name = "6h_1d_1w_elder_ray_volume_pivot_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1w EMA(13) for trend filter
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_1d_aligned  # High - EMA13
    bear_power = ema_13_1d_aligned - low   # EMA13 - Low
    
    # Slope of power (current - previous)
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(ema_13_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filters: price relative to 1d and 1w EMA13
        is_1d_uptrend = close[i] > ema_13_1d_aligned[i]
        is_1d_downtrend = close[i] < ema_13_1d_aligned[i]
        is_1w_uptrend = close[i] > ema_13_1w_aligned[i]
        is_1w_downtrend = close[i] < ema_13_1w_aligned[i]
        
        # Entry conditions
        long_entry = (bull_power[i] > 0) and (bull_power_slope[i] > 0) and \
                     volume_filter and is_1d_uptrend and is_1w_uptrend
        short_entry = (bear_power[i] > 0) and (bear_power_slope[i] > 0) and \
                      volume_filter and is_1d_downtrend and is_1w_downtrend
        
        # Exit conditions
        long_exit = (bull_power[i] <= 0) or (bull_power_slope[i] <= 0) or \
                    (not is_1d_uptrend) or (not is_1w_uptrend)
        short_exit = (bear_power[i] <= 0) or (bear_power_slope[i] <= 0) or \
                     (not is_1d_downtrend) or (not is_1w_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals