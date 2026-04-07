#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly Donchian Breakout with Volume Filter
# Hypothesis: Weekly Donchian(20) breakouts with volume confirmation capture strong momentum moves.
# Weekly trend filter (price above/below weekly SMA20) avoids counter-trend trades.
# Volume > 1.5x 20-period average reduces false breakouts.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "4h_weekly_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly close for trend filter
    weekly_close = df_weekly['close'].values
    # Weekly high/low for Donchian channels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly SMA20 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    weekly_sma20 = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Donchian channels (20-period high/low)
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_high_20 = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_20 = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to 4h timeframe
    weekly_sma20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma20)
    weekly_high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    weekly_low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)
    
    # Volume filter on 4h: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(weekly_sma20_aligned[i]) or np.isnan(weekly_high_20_aligned[i]) or
            np.isnan(weekly_low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below weekly Donchian low
            if close[i] < weekly_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above weekly Donchian high
            if close[i] > weekly_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price above weekly SMA20 (uptrend) and breakout above weekly high with volume
            if (close[i] > weekly_sma20_aligned[i] and
                high[i] > weekly_high_20_aligned[i] and
                close[i] > weekly_high_20_aligned[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below weekly SMA20 (downtrend) and breakdown below weekly low with volume
            elif (close[i] < weekly_sma20_aligned[i] and
                  low[i] < weekly_low_20_aligned[i] and
                  close[i] < weekly_low_20_aligned[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals