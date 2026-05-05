#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 6h Donchian(20) high AND weekly close > weekly open (bullish weekly candle) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 6h Donchian(20) low AND weekly close < weekly open (bearish weekly candle) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 6h Donchian(10) midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Donchian channels provide adaptive support/resistance; weekly candle direction filters for higher timeframe momentum; volume confirms breakout validity
# Works in bull markets (breakouts with bullish weekly) and bear markets (breakdowns with bearish weekly)

name = "6h_Donchian20_WeeklyCandle_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weekly candles
        return np.zeros(n)
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Weekly bullish/bearish candle: 1 if close > open, -1 if close < open, 0 otherwise
    weekly_bullish = np.where(weekly_close > weekly_open, 1, 
                             np.where(weekly_close < weekly_open, -1, 0))
    weekly_direction_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    # Calculate 6h Donchian(20) channels
    # We need to calculate rolling max/min on the 6h data itself
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (donchian_high_20 + donchian_low_20) / 2  # midpoint for exit
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(weekly_direction_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high, weekly bullish, volume confirmation
            if close[i] > donchian_high_20[i] and weekly_direction_aligned[i] > 0 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian(20) low, weekly bearish, volume confirmation
            elif close[i] < donchian_low_20[i] and weekly_direction_aligned[i] < 0 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian(20) midpoint OR volume drops below average
            if close[i] < donchian_mid_10[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian(20) midpoint OR volume drops below average
            if close[i] > donchian_mid_10[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals