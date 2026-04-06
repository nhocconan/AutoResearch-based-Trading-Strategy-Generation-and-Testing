#!/usr/bin/env python3
"""
1h 4h Donchian Breakout + Volume Confirmation + Session Filter
Hypothesis: 4h Donchian breakouts provide strong directional signals in both bull and bear markets.
Volume confirmation ensures breakouts are genuine, not false moves. Session filter (08-20 UTC)
focuses on active London/New York overlap, reducing noise. 1h timeframe provides precise entry
timing while 4h trend direction controls trade frequency. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14414_1h_donchian4h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_high = high_4h_series.rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = low_4h_series.rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # 4h volume moving average for confirmation
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Volume filter: avoid low volume periods
    volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (0.8 * volume_ma_1h)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (London/NY overlap)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_len, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_4h_aligned[i]) or np.isnan(volume_ma_1h[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20  # maintain position outside session
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below 4h Donchian low OR stoploss
            if (close[i] <= donchian_low_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above 4h Donchian high OR stoploss
            if (close[i] >= donchian_high_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 4h Donchian breakout + volume confirmation
            # Volume confirmation: current 4h volume > 20-period average
            volume_confirmed = volume_4h[i] > volume_ma_4h_aligned[i]
            
            long_setup = (close[i] > donchian_high_aligned[i] and volume_confirmed)
            short_setup = (close[i] < donchian_low_aligned[i] and volume_confirmed)
            
            if long_setup:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals