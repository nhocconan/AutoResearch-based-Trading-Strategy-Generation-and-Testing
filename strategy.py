#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v2
# Hypothesis: 6h strategy using weekly pivot points (from 1w data) for trend direction and daily Donchian breakouts (20-period) for entry timing.
# Long: Price breaks above 20-period Donchian high AND weekly pivot shows bullish bias (close > weekly pivot) with volume confirmation.
# Short: Price breaks below 20-period Donchian low AND weekly pivot shows bearish bias (close < weekly pivot) with volume confirmation.
# Exit: Opposite Donchian breakout (long exits on Donchian low break, short exits on Donchian high break).
# Volume confirmation: current volume > 1.5x 20-period average to filter low-momentum breakouts.
# Weekly pivot provides higher-timeframe trend filter to avoid counter-trend trades in ranging markets.
# Donchian breakouts capture momentum moves; weekly pivot ensures alignment with major trend.
# Target: 15-30 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low (opposite breakout)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (opposite breakout)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Donchian high breakout + weekly pivot bullish + volume
            if (close[i] > donchian_high[i] and    # Break above Donchian high
                close[i] > pivot_1w_aligned[i] and  # Price above weekly pivot (bullish bias)
                volume_confirmed):                  # Volume spike
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian low breakout + weekly pivot bearish + volume
            elif (close[i] < donchian_low[i] and   # Break below Donchian low
                  close[i] < pivot_1w_aligned[i] and  # Price below weekly pivot (bearish bias)
                  volume_confirmed):                  # Volume spike
                position = -1
                signals[i] = -0.25
    
    return signals