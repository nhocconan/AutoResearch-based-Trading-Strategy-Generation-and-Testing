#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_v1
4-hour strategy combining Donchian channel breakout with volume confirmation and trend filter.
Enters long when price breaks above 4-period high on daily chart with volume confirmation.
Enters short when price breaks below 4-period low on daily chart with volume confirmation.
Uses daily ADX to filter ranging markets (ADX < 20).
Target: 20-50 trades per year per symbol.
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
    
    # === Daily Donchian Channels (4-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4-period high and low channels
    high_4 = pd.Series(high_1d).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low_1d).rolling(window=4, min_periods=4).min().values
    
    # Align to 4h timeframe (wait for daily close)
    high_4_aligned = align_htf_to_ltf(prices, df_1d, high_4)
    low_4_aligned = align_htf_to_ltf(prices, df_1d, low_4)
    
    # === Daily Volume Confirmation (10-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === Daily ADX for Trend Filter (trending only) ===
    # Calculate True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_4_aligned[i]) or 
            np.isnan(low_4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current day's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.3 * vol_ma_1d_aligned[i]
        
        # Trend filter: only trade in trending markets (ADX > 20)
        trending = adx_aligned[i] > 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 4-day high with volume confirmation and trending
            if (close[i] > high_4_aligned[i] and vol_confirmed and trending):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 4-day low with volume confirmation and trending
            elif (close[i] < low_4_aligned[i] and vol_confirmed and trending):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite channel level
        elif position == 1:
            # Exit long: price crosses below 4-day low
            if close[i] < low_4_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 4-day high
            if close[i] > high_4_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0