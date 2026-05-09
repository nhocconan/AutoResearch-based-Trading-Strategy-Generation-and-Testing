#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 20-period Donchian breakout with 12h EMA50 trend filter and volume spike.
# Uses 12h EMA for trend direction, Donchian channels for breakout signals,
# and volume surge for confirmation. Designed to work in both bull (breakouts above upper channel)
# and bear (breakdowns below lower channel). Target: 20-50 trades/year to avoid fee drag.
name = "4h_Donchian20_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 50-period EMA for 12h timeframe
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) for 4h timeframe
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # For true Donchian, we need to reset the accumulation every 20 periods
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i < 20:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
        else:
            upper_channel[i] = np.max(high[i-19:i+1])
            lower_channel[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: volume > 2.0x 20-period EMA (strict threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + 12h EMA50 > price + volume spike
            if (price > upper_channel[i] and ema_50_12h_aligned[i] > price and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel + 12h EMA50 < price + volume spike
            elif (price < lower_channel[i] and ema_50_12h_aligned[i] < price and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below upper channel or 12h EMA50 drops below price
            if price < upper_channel[i] or ema_50_12h_aligned[i] < price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above lower channel or 12h EMA50 rises above price
            if price > lower_channel[i] or ema_50_12h_aligned[i] > price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals