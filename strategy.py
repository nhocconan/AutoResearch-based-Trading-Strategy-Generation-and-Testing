#!/usr/bin/env python3
"""
12h Donchian Channel Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms breakout strength.
ADX ensures we only trade in trending markets (ADX > 25), avoiding whipsaws in ranging conditions.
Works in bull markets (breakouts to upside) and bear markets (breakouts to downside).
Target: 100-180 total trades over 4 years (25-45/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14452_12h_donchian20_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX and Donchian channels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    adx_period = 14
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Smooth TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Donchian Channel (20-period)
    donch_period = 20
    upper = pd.Series(high_1d).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low_1d).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: require volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # 20% above average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(adx_period, donch_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR trend weakens OR stoploss
            if (close[i] < lower_aligned[i] or not trending or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR trend weakens OR stoploss
            if (close[i] > upper_aligned[i] or not trending or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + trend
            long_breakout = close[i] > upper_aligned[i]
            short_breakout = close[i] < lower_aligned[i]
            
            if long_breakout and vol_filter[i] and trending:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and trending:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals