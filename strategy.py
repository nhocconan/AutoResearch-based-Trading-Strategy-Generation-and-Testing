#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian channel breakout and volume confirmation.
Trade breakouts above 20-period 1d high (long) or below 20-period 1d low (short).
Use volume > 2.0x 20-period 1d average volume to confirm breakout strength.
Use 1d ADX > 20 to filter for trending markets and avoid false breakouts in ranging markets.
Position sizing: 0.25 for entries, 0 for exits.
Target: 75-200 total trades over 4 years (19-50/year).
Donchian channels from 1d provide robust structure levels that adapt to volatility and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from ADX components
        uptrend = plus_di_aligned[i] > minus_di_aligned[i]
        downtrend = plus_di_aligned[i] < minus_di_aligned[i]
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, strong trend
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 2.0 and 
                strong_trend and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, strong trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 2.0 and 
                  strong_trend and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or trend weakens
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or trend weakens
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0