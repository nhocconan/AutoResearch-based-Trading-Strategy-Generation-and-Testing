#!/usr/bin/env python3
"""
12h Donchian Channel Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation ensures
breakout validity, while ADX filter avoids choppy markets. Works in bull (buy breakouts)
and bear (sell breakdowns) by trading both directions. Target: 100-180 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14432_12h_donchian20_vol_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel parameters
    donchian_period = 20
    
    # Donchian Upper Channel: highest high over N periods
    donchian_high = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    # Donchian Lower Channel: lowest low over N periods
    donchian_low = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ADX calculation for trend strength
    adx_period = 14
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: require volume above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
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
    start = max(donchian_period, adx_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trending (ADX > 20)
        trending = adx_aligned[i] > 20
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR trend weakens OR stoploss
            if (close[i] <= donchian_low_aligned[i] or not trending or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR trend weakens OR stoploss
            if (close[i] >= donchian_high_aligned[i] or not trending or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + trend filter
            long_breakout = close[i] > donchian_high_aligned[i]
            short_breakout = close[i] < donchian_low_aligned[i]
            
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