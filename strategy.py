#!/usr/bin/env python3
"""
1d Donchian(20) breakout + Volume confirmation + ATR stoploss
Hypothesis: Donchian breakouts capture momentum in trending markets while volume confirmation filters false breakouts.
ATR-based stoploss manages risk. Works in bull markets (breakout long) and bear markets (breakdown short).
Target: 50-150 total trades over 4 years (12-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14364_1d_donchian20_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Donchian channel parameters
    donchian_period = 20
    
    # 1d data for weekly context (optional but good practice)
    df_1w = get_htf_data(prices, '1w')
    
    # Price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: upper = rolling max(high), lower = rolling min(low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: avoid low volume breakouts
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5  # Require 150% of average volume for breakout
    vol_filter = volume > (vol_threshold * vol_ma)
    
    # ATR for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start = donchian_period
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits for existing positions
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR stoploss
            if (close[i] <= donchian_lower[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR stoploss
            if (close[i] >= donchian_upper[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for new entries: Donchian breakout + volume confirmation
            long_breakout = (close[i] > donchian_upper[i]) and vol_filter[i]
            short_breakout = (close[i] < donchian_lower[i]) and vol_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals