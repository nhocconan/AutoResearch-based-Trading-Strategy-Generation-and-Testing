#!/usr/bin/env python3
"""
6h_Aggressive_Trend_Filter_v1
Aggressive trend-following strategy for 6h timeframe using 20-period Donchian breakout 
combined with 50-period EMA filter and volume confirmation. Uses 1d ADX(14) > 25 as 
trend strength filter from higher timeframe. Designed to capture strong trends while 
avoiding choppy markets in both bull and bear regimes.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 20-period Donchian channels for breakout ===
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 50-period EMA for trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ADX(14) for higher timeframe trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high20[i]) or 
            np.isnan(low20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 20-period high, price above EMA50, ADX > 25, volume confirmed
            if (close[i] > high20[i] and 
                close[i] > ema50[i] and 
                adx_14_1d_aligned[i] > 25 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 20-period low, price below EMA50, ADX > 25, volume confirmed
            elif (close[i] < low20[i] and 
                  close[i] < ema50[i] and 
                  adx_14_1d_aligned[i] > 25 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: trend weakening or reversal
        elif position == 1:
            # Exit long: price closes below EMA50 OR ADX drops below 20
            if (close[i] < ema50[i] or 
                adx_14_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above EMA50 OR ADX drops below 20
            if (close[i] > ema50[i] or 
                adx_14_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Aggressive_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0