#!/usr/bin/env python3
"""
4h_RangeBreakout_Volume_V1
Range breakout with volume confirmation and trend filter.
Long when price breaks above 20-bar high + volume spike + trend up.
Short when price breaks below 20-bar low + volume spike + trend down.
Exit when price returns to 10-bar moving average.
Uses 1d ADX for trend filter to avoid whipsaws.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 20-bar high/low for breakout ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 10-bar mean for exit ===
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # === Volume spike (2x 20-bar average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # === 1d ADX for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on daily data
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range
    tr1 = d_high - d_low
    tr2 = np.abs(d_high - np.roll(d_close, 1))
    tr3 = np.abs(d_low - np.roll(d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where((d_high[1:] - d_high[:-1]) > (d_low[:-1] - d_low[1:]), 
                       np.maximum(d_high[1:] - d_high[:-1], 0), 0)
    minus_dm = np.where((d_low[:-1] - d_low[1:]) > (d_high[1:] - d_high[:-1]), 
                        np.maximum(d_low[:-1] - d_low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth with Wilder's smoothing (using EMA as approximation)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d * 14)
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d * 14)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ma_10[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above 20-bar high + volume spike + ADX > 20
            if (close[i] > high_20[i] and 
                volume_spike[i] and 
                adx_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below 20-bar low + volume spike + ADX > 20
            elif (close[i] < low_20[i] and 
                  volume_spike[i] and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to 10-bar MA
            if close[i] <= ma_10[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-bar MA
            if close[i] >= ma_10[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RangeBreakout_Volume_V1"
timeframe = "4h"
leverage = 1.0