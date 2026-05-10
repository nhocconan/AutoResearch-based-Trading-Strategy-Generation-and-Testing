#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels from 1d provide strong support/resistance. Breakout above R3 or below S3 with 1d trend alignment (price > EMA34) and volume confirmation (volume > 1.5x EMA20) captures strong moves in both bull and bear markets. Low trade frequency expected due to multiple confluence filters.
# Target: 20-50 trades/year on 4h timeframe.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    pp = (high + low + close) / 3
    range_hl = high - low
    r3 = close + range_hl * 1.1 / 2
    s3 = close - range_hl * 1.1 / 2
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily timeframe (using previous day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data for today's levels (no look-ahead)
    r3_1d, s3_1d = calculate_camarilla(
        np.roll(high_1d, 1), 
        np.roll(low_1d, 1), 
        np.roll(close_1d, 1)
    )
    # First day has no previous data, set to NaN
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Calculate EMA34 on daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 4h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1) + EMA34 (34) + vol EMA (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price above EMA34 (uptrend) AND volume filter
            if close[i] > r3_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price below EMA34 (downtrend) AND volume filter
            elif close[i] < s3_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 or EMA34
            if close[i] < s3_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or EMA34
            if close[i] > r3_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals