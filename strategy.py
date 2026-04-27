#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and volume spike.
# Donchian breakout captures breakouts in trending markets.
# ADX > 25 on 1d confirms strong trend to avoid false breakouts in ranging markets.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Works in both bull and bear markets by filtering breakouts with trend strength.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.concatenate([[0], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[0], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, period):
        result = np.full_like(val, np.nan, dtype=float)
        if len(val) < period:
            return result
        result[period-1] = np.nansum(val[:period])
        for i in range(period, len(val)):
            result[i] = result[i-1] - (result[i-1] / period) + val[i]
        return result
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX > 25 indicates strong trend
        if adx_1d_aligned[i] > 25 and volume_filter[i]:
            # Long breakout: price breaks above upper Donchian channel
            if close[i] > high_20[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian channel
            elif close[i] < low_20[i]:
                signals[i] = -0.25
                position = -1
            # Exit conditions: reverse signal or loss of momentum
            elif position == 1 and close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            # Hold position if still in trend
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # No strong trend or no volume: stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_Donchian20_1dADX25_VolumeFilter"
timeframe = "6h"
leverage = 1.0