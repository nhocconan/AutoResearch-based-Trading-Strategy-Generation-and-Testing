#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume_Filter
Hypothesis: 4h Camarilla pivot breakouts with 12h volume confirmation and 12h ADX trend filter.
Camarilla levels provide high-probability reversal/breakout points. Volume confirms institutional interest.
ADX ensures we only trade in trending markets, avoiding whipsaws in ranges.
Designed for 4h timeframe with target 20-50 trades/year to minimize fee drag.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla multipliers
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    r2 = close_1d + rng * 1.1 / 6
    r3 = close_1d + rng * 1.1 / 4
    r4 = close_1d + rng * 1.1 / 2
    s1 = close_1d - rng * 1.1 / 12
    s2 = close_1d - rng * 1.1 / 6
    s3 = close_1d - rng * 1.1 / 4
    s4 = close_1d - rng * 1.1 / 2
    
    # Combine all levels
    camarilla_levels = np.column_stack([r4, r3, r2, r1, s1, s2, s3, s4])
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_aligned = align_htf_to_ltf(prices, df_1d, camarilla_levels)
    
    # Load 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    up_sum = pd.Series(up_move).rolling(window=14, min_periods=14).sum().values
    down_sum = pd.Series(down_move).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * up_sum / tr_sum
    minus_di = 100 * down_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h indicators to 4h timeframe
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any data is invalid
        if (np.isnan(vol_avg_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_aligned[i, 0]) or np.isnan(camarilla_aligned[i, 7])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current Camarilla levels (from previous day)
        r4, r3, r2, r1, s1, s2, s3, s4 = camarilla_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > vol_avg_aligned[i] * 1.5
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        # Long: price breaks above R4 with volume and trend
        long_breakout = (close[i] > r4) and volume_confirm and strong_trend
        # Short: price breaks below S4 with volume and trend
        short_breakout = (close[i] < s4) and volume_confirm and strong_trend
        
        # Exit conditions: opposite breakout or loss of trend/volume
        long_exit = (close[i] < r3) or (adx_aligned[i] < 20) or (volume[i] < vol_avg_aligned[i] * 0.8)
        short_exit = (close[i] > s3) or (adx_aligned[i] < 20) or (volume[i] < vol_avg_aligned[i] * 0.8)
        
        # Priority: entry > exit > hold
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals