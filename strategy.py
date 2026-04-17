#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX(14) trend filter.
Long when price breaks above Donchian upper with volume > 1.5x 1d average volume AND 1d ADX > 25.
Short when price breaks below Donchian lower with volume > 1.5x 1d average volume AND 1d ADX > 25.
Exit when price touches the opposite Donchian level.
Designed to capture strong trends with volume confirmation, avoiding choppy markets via ADX filter.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
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
    
    # Get 1d data for volume MA and ADX
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA(20)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1d ADX(14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[0:13] = np.nan  # Warmup period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Donchian(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: 1d ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_upper = close[i] > donch_high[i]
        breakout_lower = close[i] < donch_low[i]
        
        if position == 0:
            # Long: break above upper with volume confirmation and ADX > 25
            if (breakout_upper and volume_confirmed and trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with volume confirmation and ADX > 25
            elif (breakout_lower and volume_confirmed and trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower Donchian
            if close[i] <= donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper Donchian
            if close[i] >= donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dADX_Trend"
timeframe = "4h"
leverage = 1.0