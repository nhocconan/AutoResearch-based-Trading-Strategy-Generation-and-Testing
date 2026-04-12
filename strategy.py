#!/usr/bin/env python3
"""
6h_1d_1w_Camarilla_Range_Bound_v1
Hypothesis: On 6h timeframe, fade at Camarilla R3/S3 levels from daily pivot when weekly trend is weak (ADX < 25),
and trade breakouts at R4/S4 when weekly trend is strong (ADX > 25). Uses volume confirmation to avoid false signals.
Designed for 15-35 trades/year by requiring confluence: price at Camarilla extreme, weekly ADX regime filter, volume spike.
Works in ranging markets via mean reversion at R3/S3 and in trending markets via breakout continuation at R4/S4.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Camarilla_Range_Bound_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots: (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Weekly ADX for trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).fillna(0).values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).fillna(0).values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).fillna(0).values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).fillna(0).values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (24 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price proximity to Camarilla levels (within 0.3%)
        near_r3 = abs(close[i] - r3_1d_aligned[i]) / r3_1d_aligned[i] <= 0.003
        near_s3 = abs(close[i] - s3_1d_aligned[i]) / s3_1d_aligned[i] <= 0.003
        near_r4 = abs(close[i] - r4_1d_aligned[i]) / r4_1d_aligned[i] <= 0.003
        near_s4 = abs(close[i] - s4_1d_aligned[i]) / s4_1d_aligned[i] <= 0.003
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Weekly ADX regime
        weak_trend = adx_aligned[i] < 25  # ranging market
        strong_trend = adx_aligned[i] > 25  # trending market
        
        # Entry conditions
        # In weak trend: fade at R3/S3 (mean reversion)
        long_entry = near_r3 and weak_trend and volume_spike
        short_entry = near_s3 and weak_trend and volume_spike
        
        # In strong trend: breakout continuation at R4/S4
        long_breakout = near_r4 and strong_trend and volume_spike and close[i] > close[i-1]
        short_breakout = near_s4 and strong_trend and volume_spike and close[i] < close[i-1]
        
        # Exit conditions: price moves to opposite level or ADX regime changes
        long_exit = near_s3 or (adx_aligned[i] < 20 and position == 1)  # exit to S3 or trend weakening
        short_exit = near_r3 or (adx_aligned[i] < 20 and position == -1)  # exit to R3 or trend weakening
        
        # Priority: entry > exit > hold
        if (long_entry or long_breakout) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_entry or short_breakout) and position != -1:
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