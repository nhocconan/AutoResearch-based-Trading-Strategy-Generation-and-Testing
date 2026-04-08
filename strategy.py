#!/usr/bin/env python3
# 6h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: Donchian(20) breakout on 6h timeframe with 12h EMA200 trend filter and volume confirmation.
# Works in both bull and bear markets by following the higher timeframe trend while capturing breakouts.
# Target: 15-30 trades/year via strict breakout conditions + trend alignment + volume filter.

name = "6h_donchian_breakout_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA200 for higher timeframe trend
    ema200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(200, 20)  # Need enough data for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get aligned 12h EMA200 for trend filter
        ema200_12h_val = align_htf_to_ltf(prices, df_12h, ema200_12h)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema200_12h_val) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price closes below Donchian lower OR trend changes
            if close[i] < low_20[i] or ema200_12h_val >= close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above Donchian upper OR trend changes
            if close[i] > high_20[i] or ema200_12h_val <= close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian upper with volume, aligned with uptrend
            if (close[i] > high_20[i] and 
                volume_filter and 
                ema200_12h_val < close[i]):  # Price above 12h EMA200 = uptrend
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian lower with volume, aligned with downtrend
            elif (close[i] < low_20[i] and 
                  volume_filter and 
                  ema200_12h_val > close[i]):  # Price below 12h EMA200 = downtrend
                position = -1
                signals[i] = -0.25
    
    return signals