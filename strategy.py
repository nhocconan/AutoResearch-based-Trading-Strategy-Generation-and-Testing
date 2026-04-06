#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, weekly trend up (price > weekly EMA(20)), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, weekly trend down (price < weekly EMA(20)), volume > 1.5x avg
# Exit when: price reverses to Donchian(10) opposite side OR weekly trend flips
# Uses weekly trend to filter breakouts, targeting 30-80 trades over 4 years

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < Donchian(10) low OR weekly trend flips down
            donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().iloc[i]
            if close[i] < donchian_low_10 or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > Donchian(10) high OR weekly trend flips up
            donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().iloc[i]
            if close[i] > donchian_high_10 or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price breaks Donchian(20) + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i]:
                    # Breakout above weekly EMA - bullish
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i]:
                    # Breakdown below weekly EMA - bearish
                    signals[i] = -0.25
                    position = -1
    
    return signals