#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h strategy using Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long: Price breaks above 6h Donchian(20) high with volume > 1.5x 20-period average and weekly pivot > weekly open (bullish bias).
# Short: Price breaks below 6h Donchian(20) low with volume > 1.5x 20-period average and weekly pivot < weekly open (bearish bias).
# Exit: Price returns to opposite Donchian level (long exits below Donchian low, short exits above Donchian high).
# Uses weekly pivot from 1w data for directional filter: only long when weekly pivot > weekly open, only short when weekly pivot < weekly open.
# Target: 12-37 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.
# Donchian channels provide clear breakout levels, weekly pivot adds higher-timeframe bias, volume confirms conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
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
    open_prices = prices['open'].values
    
    # Donchian(20) on 6h data
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot point: (high + low + close) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot and open to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    open_1w_aligned = align_htf_to_ltf(prices, df_1w, open_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(open_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Weekly bullish bias: pivot > weekly open
        weekly_bullish = weekly_pivot_aligned[i] > open_1w_aligned[i]
        # Weekly bearish bias: pivot < weekly open
        weekly_bearish = weekly_pivot_aligned[i] < open_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high with volume and weekly bullish bias
            if (close[i] > donchian_high[i] and    # Break above Donchian high
                volume_confirmed and               # Volume spike
                weekly_bullish):                   # Weekly bullish bias
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low with volume and weekly bearish bias
            elif (close[i] < donchian_low[i] and   # Break below Donchian low
                  volume_confirmed and             # Volume spike
                  weekly_bearish):                 # Weekly bearish bias
                position = -1
                signals[i] = -0.25
    
    return signals