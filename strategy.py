#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume Filter
# Hypothesis: 20-period Donchian channel breakouts on 4h chart capture strong trends.
# Volume filter ensures only institutional participation triggers entries.
# Works in both bull and bear markets: In bull, breaks above upper channel continue up; in bear, breaks below lower channel continue down.
# Uses 1d timeframe for volume average to avoid noise.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian_breakout_volume_v1"
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
    
    # Get daily data for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_daily_series = pd.Series(vol_daily)
    vol_ma_daily = vol_daily_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_daily_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or volume drops below average
            if close[i] <= donchian_low[i] or volume[i] < vol_ma_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or volume drops below average
            if close[i] >= donchian_high[i] or volume[i] < vol_ma_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume above average
            if close[i] > donchian_high[i] and volume[i] > vol_ma_daily_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume above average
            elif close[i] < donchian_low[i] and volume[i] > vol_ma_daily_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals