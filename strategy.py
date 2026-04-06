#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Regime Filter
Hypothesis: Price breakouts from Donchian channels with volume confirmation and ADX trend filter capture strong moves in both bull and bear markets. The 4h timeframe reduces trade frequency to avoid fee drag, while volume and ADX ensure trades align with institutional participation and trending conditions. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_adx"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 4h data)
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h close for price action (aligned to 1h timeframe if needed, but we're using 4h as primary)
    # Since timeframe is 4h, we need to align 4h data to 4h bars
    # For 4h primary timeframe, we can use the 4h data directly with proper indexing
    close = close_4h
    high = high_4h
    low = low_4h
    volume = volume_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 14 for ATR, etc.)
    start = 30
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR stoploss
            if (close[i] <= lowest_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= highest_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX trend filter
            # Long: price breaks above upper Donchian band with high volume and strong uptrend (ADX > 25)
            long_breakout = close[i] > highest_high[i]
            long_volume = volume[i] > 1.5 * vol_ma[i]
            strong_uptrend = adx_aligned[i] > 25
            
            # Short: price breaks below lower Donchian band with high volume and strong downtrend
            short_breakout = close[i] < lowest_low[i]
            short_volume = volume[i] > 1.5 * vol_ma[i]
            strong_downtrend = adx_aligned[i] > 25
            
            if long_breakout and long_volume and strong_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and short_volume and strong_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals