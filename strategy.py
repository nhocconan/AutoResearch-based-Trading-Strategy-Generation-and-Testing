#!/usr/bin/env python3
"""
12h CAMARILLA PIVOT BREAKOUT WITH VOLUME AND TREND FILTER
Uses daily CAMARILLA pivot levels from prior day for breakout signals.
Only takes longs above H3 and shorts below L3 when:
1. Price breaks pivot level with close
2. Volume > 1.5x 20-period average (volume confirmation)
3. 1-day ADX > 25 (trending market filter)
Position size: 0.25. Designed for 12h timeframe targeting 50-150 total trades.
Works in bull/bear by following trend direction via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for CAMARILLA pivots and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate CAMARILLA pivot levels from prior day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # CAMARILLA levels (based on prior day)
    # H4 = C + 1.1 * (H - L)
    # H3 = C + 1.1 * (H - L) / 2
    # H2 = C + 1.1 * (H - L) / 4
    # H1 = C + 1.1 * (H - L) / 6
    # L1 = C - 1.1 * (H - L) / 6
    # L2 = C - 1.1 * (H - L) / 4
    # L3 = C - 1.1 * (H - L) / 2
    # L4 = C - 1.1 * (H - L)
    
    # We use H3 for long entries, L3 for short entries
    h3 = close_1d + 1.1 * range_1d / 2
    l3 = close_1d - 1.1 * range_1d / 2
    
    # Align CAMARILLA levels to 12h timeframe (use prior day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # 1-day ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr0 = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) > 0, dx, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX > 25 = trending market
    trending = adx > 25
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: CAMARILLA breakout + volume spike + trending market
        long_entry = (close[i] > h3_aligned[i] and 
                      vol_spike_aligned[i] > 0.5 and 
                      trending_aligned[i] > 0.5)
        short_entry = (close[i] < l3_aligned[i] and 
                       vol_spike_aligned[i] > 0.5 and 
                       trending_aligned[i] > 0.5)
        
        # Exit when price returns to opposite H3/L3 level or close touches pivot
        # For long: exit if price drops back below H3
        # For short: exit if price rises back above L3
        exit_long = position == 1 and close[i] <= h3_aligned[i]
        exit_short = position == -1 and close[i] >= l3_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_camarilla_pivot_breakout_volume_trend"
timeframe = "12h"
leverage = 1.0