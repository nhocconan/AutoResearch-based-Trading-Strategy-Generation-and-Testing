#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 15-period Donchian breakout with weekly trend filter and volume confirmation.
# Donchian channels provide clear breakout levels; weekly trend ensures directional bias;
# volume confirmation reduces false breakouts. Designed for low trade frequency (<25/year)
# to avoid fee drag, works in bull/bear via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    
    # Calculate weekly EMA(40) for trend
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, weekly, weekly_ema)
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 1d volume > 1.5x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above Donchian high AND above weekly EMA (uptrend)
            if close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND below weekly EMA (downtrend)
            elif close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian20_WeeklyEMA40_VolumeFilter"
timeframe = "1d"
leverage = 1.0