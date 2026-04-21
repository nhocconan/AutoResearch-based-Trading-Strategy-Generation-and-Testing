#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Donchian channel breakouts on 4h timeframe with volume confirmation and daily trend filter provide edge in both bull and bear markets. Uses 20-period Donchian channels for breakout detection, volume > 1.5x 20-period average for confirmation, and daily EMA50 for trend filter. Designed for low trade frequency (target: 20-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.full_like(volume, np.nan)
    for i in range(20, n):
        volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        trend_filter = ema50_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation and uptrend
            if price > upper_channel and vol_ok and price > trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume confirmation and downtrend
            elif price < lower_channel and vol_ok and price < trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle of channel or breaks below lower channel
            middle_channel = (upper_channel + lower_channel) / 2.0
            if price < middle_channel or price < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle of channel or breaks above upper channel
            middle_channel = (upper_channel + lower_channel) / 2.0
            if price > middle_channel or price > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0