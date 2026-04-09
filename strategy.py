#!/usr/bin/env python3
# 6h_1d_price_channel_breakout_v1
# Hypothesis: 6-hour price channel breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 6h Donchian high (20) with weekly close > weekly open (bullish weekly candle) and volume > 1.8x 20-bar average.
# Short when price breaks below 6h Donchian low (20) with weekly close < weekly open (bearish weekly candle) and volume > 1.8x 20-bar average.
# Exit when price returns to the opposite Donchian level (exit long at Donchian low, exit short at Donchian high).
# Uses weekly trend to filter breakout direction, reducing false signals in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_price_channel_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly bullish/bearish candle (1 if close > open, -1 if close < open)
    weekly_bullish = np.where(df_weekly['close'].values > df_weekly['open'].values, 1, -1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            window_high = high[i-19:i+1]
            window_low = low[i-19:i+1]
            donchian_high[i] = np.max(window_high)
            donchian_low[i] = np.min(window_low)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with bullish weekly and volume
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] == 1 and 
                volume[i] > vol_ma_20[i] * 1.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with bearish weekly and volume
            elif (close[i] < donchian_low[i] and 
                  weekly_bullish_aligned[i] == -1 and 
                  volume[i] > vol_ma_20[i] * 1.8):
                position = -1
                signals[i] = -0.25
    
    return signals