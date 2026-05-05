#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d EMA200) and volume spike confirmation
# Long when price breaks above 6h Donchian upper band AND price > 1d EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 6h Donchian lower band AND price < 1d EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 6h Donchian midpoint (mean of upper/lower band) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Donchian channels provide robust trend-following structure; 1d EMA200 filters major trend direction; volume confirms breakout strength
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "6h_Donchian20_1dEMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough for 6h Donchian(20) and 1d EMA200
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need enough for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period (max of 200 for EMA and 20 for Donchian)
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band, above 1d EMA200, volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band, below 1d EMA200, volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian middle OR volume drops below average
            if close[i] < donchian_middle[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian middle OR volume drops below average
            if close[i] > donchian_middle[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals