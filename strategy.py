#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > weekly pivot (PP), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < weekly pivot (PP), volume > 1.5x avg
# Weekly pivot (PP) from 1d data: (weekly high + weekly low + weekly close) / 3
# Exit when: price returns to weekly pivot (PP) or opposite Donchian band is touched
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull (breakouts) and bear (fades at pivot)

name = "6h_donchian20_1d_weeklypivot_vol_v1"
timeframe = "6h"
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
    
    # Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot from 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Resample 1d to weekly (using actual weekly bars from data)
    # Since we have daily data, we calculate weekly pivot as:
    # Weekly high = max of last 7 days high
    # Weekly low = min of last 7 days low  
    # Weekly close = close of 7 days ago
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close_1d).shift(6).values  # close 7 days ago
    
    # Weekly pivot point (PP) = (weekly high + weekly low + weekly close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price <= weekly pivot OR price touches lower Donchian band
            if close[i] <= weekly_pivot_aligned[i] or close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price >= weekly pivot OR price touches upper Donchian band
            if close[i] >= weekly_pivot_aligned[i] or close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + pivot bias + volume
            if volume[i] > volume_threshold[i]:
                # Long: break above Donchian high AND price > weekly pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below Donchian low AND price < weekly pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals