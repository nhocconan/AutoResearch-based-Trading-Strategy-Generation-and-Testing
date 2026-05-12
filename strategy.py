#!/usr/bin/env python3
# 6h_LongTermDonchianWithVolumeFilter
# Hypothesis: Use 20-period Donchian breakout on 6h with volume confirmation and 1d trend filter.
# Enter long when price breaks above 20-period high with volume above 20-period average and 1d close > 1d EMA50.
# Enter short when price breaks below 20-period low with volume above 20-period average and 1d close < 1d EMA50.
# Exit when price returns to the 20-period midpoint. Designed for low frequency (10-25 trades/year)
# to avoid fee drag. Works in trending markets by capturing breakouts with volume confirmation,
# and avoids false breakouts in ranging markets via volume and trend filters.

name = "6h_LongTermDonchianWithVolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i - lookback + 1:i + 1])
            lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            avg_volume[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on daily
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50[i-1] * (49 / (50 + 1)))
    
    # Trend filter: 1 = bullish (close > EMA50), -1 = bearish (close < EMA50)
    trend_filter = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if not np.isnan(ema_50[i]):
            if close_1d[i] > ema_50[i]:
                trend_filter[i] = 1
            elif close_1d[i] < ema_50[i]:
                trend_filter[i] = -1
    
    # Align daily trend to 6h
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter)
    
    signals = np.zeros(n)
    
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(trend_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume above average
        volume_ok = volume[i] > avg_volume[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Midpoint for exit
        midpoint = (highest_high[i] + lowest_low[i]) / 2
        return_to_mid = abs(close[i] - midpoint) < (highest_high[i] - lowest_low[i]) * 0.1
        
        # Only trade in direction of 1d trend
        if trend_filter_aligned[i] == 1:  # Bullish trend on daily
            if breakout_up and volume_ok:
                signals[i] = 0.25  # Long
            elif return_to_mid:
                signals[i] = 0.0   # Exit long
            else:
                signals[i] = 0.0   # No signal
        elif trend_filter_aligned[i] == -1:  # Bearish trend on daily
            if breakout_down and volume_ok:
                signals[i] = -0.25  # Short
            elif return_to_mid:
                signals[i] = 0.0   # Exit short
            else:
                signals[i] = 0.0   # No signal
        else:
            signals[i] = 0.0  # No trend, stay flat
    
    return signals