#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian20 Breakout with Daily Volume Filter
# Hypothesis: Donchian(20) breakouts on 12h chart capture strong trends, validated by daily volume surge.
# Works in bull markets: breaks above upper band continue upward.
# Works in bear markets: breaks below lower band continue downward.
# Volume filter ensures only institutional participation triggers entries.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "12h_donchian20_volume_filter_v1"
timeframe = "12h"
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
    
    # Get daily data for volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align daily volume average to 12h timeframe
    daily_volume = df_daily['volume'].values
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma_aligned = align_htf_to_ltf(prices, df_daily, daily_vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or volume drops
            if (close[i] <= donchian_low[i] or 
                volume[i] <= daily_vol_ma_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or volume drops
            if (close[i] >= donchian_high[i] or 
                volume[i] <= daily_vol_ma_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume surge
            if (high[i] > donchian_high[i] and 
                volume[i] > 1.5 * daily_vol_ma_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume surge
            elif (low[i] < donchian_low[i] and 
                  volume[i] > 1.5 * daily_vol_ma_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals