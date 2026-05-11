#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3, S3) from 1d act as strong support/resistance. 
Price breaking above R3 or below S3 with 1d trend alignment (close > EMA34) and volume surge signals momentum continuation.
In ranging markets (ADX < 20), fade at R3/S3 for mean reversion. Uses 12h timeframe with 1d HTF for pivot, trend, volume, and ADX.
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
Works in bull/bear via trend filter and mean-reversion mode in ranging conditions.
"""

name = "12h_Camarilla_Pivot_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d ADX(14) for regime detection ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / (tr14 + 1e-10)
    minus_di = 100 * minus_dm14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Camarilla Pivot Levels (R3, S3) ---
    # Calculate from previous day's OHLC
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla calculations
    range_prev = prev_high - prev_low
    camarilla_mult = 1.1 / 12  # ~0.09167
    r3 = prev_close + range_prev * camarilla_mult * 4
    s3 = prev_close - range_prev * camarilla_mult * 4
    
    # Align to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                # Exit on close beyond opposite level
                if position == 1 and close[i] <= s3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= r3_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Regime detection: trending vs ranging
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.5x 1d EMA average
        vol_confirm = volume[i] > 1.5 * vol_avg_1d_aligned[i]
        
        if position == 0:
            if trending and vol_confirm:
                # Trending market: breakout continuation
                if close[i] > r3_12h[i] and close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.25  # long breakout above R3
                    position = 1
                elif close[i] < s3_12h[i] and close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.25  # short breakdown below S3
                    position = -1
            elif ranging:
                # Ranging market: mean reversion at extremes
                if close[i] < s3_12h[i]:
                    signals[i] = 0.25  # long at S3 support
                    position = 1
                elif close[i] > r3_12h[i]:
                    signals[i] = -0.25  # short at R3 resistance
                    position = -1
        else:
            # Manage existing position
            if position == 1:
                # Long: exit on close below S3 or trend reversal
                if close[i] < s3_12h[i] or (trending and close[i] < ema34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit on close above R3 or trend reversal
                if close[i] > r3_12h[i] or (trending and close[i] > ema34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals