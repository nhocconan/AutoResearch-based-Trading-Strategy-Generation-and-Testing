#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 4-hour Donchian breakout with 1-day trend filter (EMA) and volume confirmation.
# Uses 4h timeframe for breakout signals, 1-day EMA for trend direction, and volume for confirmation.
# Works in both bull and bear markets by following higher timeframe trend while capturing breakouts.
# Target: 15-30 trades/year via strict breakout conditions + trend alignment + volume filter.

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # 4-hour Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA (50-period) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, 20)  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Get aligned 1-day EMA value
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_val) or
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price closes below Donchian lower
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above Donchian upper
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above Donchian upper with volume, above 1-day EMA
            if (close[i] > high_20[i] and 
                volume_filter and 
                close[i] > ema_50_1d_val):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian lower with volume, below 1-day EMA
            elif (close[i] < low_20[i] and 
                  volume_filter and 
                  close[i] < ema_50_1d_val):
                position = -1
                signals[i] = -0.25
    
    return signals