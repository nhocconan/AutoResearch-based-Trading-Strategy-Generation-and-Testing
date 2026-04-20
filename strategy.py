#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and ADX trend filter.
# Breakouts from Donchian channels capture momentum moves. Volume confirms institutional participation.
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Works in both bull and bear markets by following the trend direction of breakouts.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_Donchian20_12hVolume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h Volume confirmation ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = vol_12h / np.where(vol_ma_12h > 0, vol_ma_12h, np.nan)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === ADX trend filter on 4h (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    plus_di = 100 * plus_dm14 / np.where(tr14 > 0, tr14, np.nan)
    minus_di = 100 * minus_dm14 / np.where(tr14 > 0, tr14, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Donchian channel breakout (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        adx_val = adx[i]
        vol_ratio_val = vol_ratio_12h_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(highest_high_val) or np.isnan(lowest_low_val) or 
            np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above 20-period high with volume and trend
            if close_val > highest_high_val and vol_ratio_val > 1.5 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below 20-period low with volume and trend
            elif close_val < lowest_low_val and vol_ratio_val > 1.5 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 20-period low or trend weakens
            if close_val < lowest_low_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 20-period high or trend weakens
            if close_val > highest_high_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals