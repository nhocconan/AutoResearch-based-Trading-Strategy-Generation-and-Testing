#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v4
# Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation.
# Long: Price breaks above Donchian(20) high AND weekly pivot > previous weekly close AND volume > 1.5x 20-period average.
# Short: Price breaks below Donchian(20) low AND weekly pivot < previous weekly close AND volume > 1.5x 20-period average.
# Exit: Opposite Donchian breakout or volume divergence.
# Uses 1w pivot for higher timeframe bias to avoid counter-trend trades in bear markets.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v4"
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
    
    # Donchian channels (20-period) for breakout
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly pivot points (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Previous weekly close for bias
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_close[0] = np.nan
    prev_weekly_close_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(prev_weekly_close_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR volume divergence (price up but volume down)
            if close[i] < donchian_low[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR volume divergence (price down but volume down)
            if close[i] > donchian_high[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, weekly pivot > prev weekly close, volume confirmed
            if (close[i] > donchian_high[i] and weekly_pivot_aligned[i] > prev_weekly_close_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, weekly pivot < prev weekly close, volume confirmed
            elif (close[i] < donchian_low[i] and weekly_pivot_aligned[i] < prev_weekly_close_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals