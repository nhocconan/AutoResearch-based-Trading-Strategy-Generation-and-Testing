#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1d weekly pivot shows bullish bias (price > weekly pivot) AND volume spike
# Short when price breaks below Donchian(20) low AND 1d weekly pivot shows bearish bias (price < weekly pivot) AND volume spike
# Weekly pivot from 1d data provides institutional reference points; Donchian breakout captures momentum; volume confirms participation
# Works in bull (breakouts with buying pressure) and bear (breakdowns with selling pressure)
# Timeframe: 6h (primary timeframe as required)
# Target: 80-160 total trades over 4 years (20-40/year) to balance signal quality and fee drag

name = "6h_Donchian20_1dWeeklyPivot_Direction_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot points (using prior week's high, low, close)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 5 trading days to get prior week's OHLC (approximation for daily data)
    # Prior week high = max of high[-10:-5], prior week low = min of low[-10:-5], prior week close = close[-5]
    prior_week_high = np.full(len(high_1d), np.nan)
    prior_week_low = np.full(len(low_1d), np.nan)
    prior_week_close = np.full(len(close_1d), np.nan)
    
    for i in range(5, len(high_1d)):
        if i >= 10:
            prior_week_high[i] = np.max(high_1d[i-10:i-5])
            prior_week_low[i] = np.min(low_1d[i-10:i-5])
            prior_week_close[i] = close_1d[i-5]
        else:
            # Not enough data for full prior week, use available
            prior_week_high[i] = np.max(high_1d[max(0, i-10):i-5]) if i >= 5 else np.nan
            prior_week_low[i] = np.min(low_1d[max(0, i-10):i-5]) if i >= 5 else np.nan
            prior_week_close[i] = close_1d[max(0, i-5)] if i >= 5 else np.nan
    
    # Weekly pivot point
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > Donchian high (breakout) AND price > weekly pivot (bullish bias) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < Donchian low (breakdown) AND price < weekly pivot (bearish bias) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price < Donchian low (mean reversion) OR loss of bullish bias
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price > Donchian high (mean reversion) OR loss of bearish bias
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals