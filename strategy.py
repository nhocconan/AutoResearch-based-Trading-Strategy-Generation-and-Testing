#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6-hour Donchian(20) breakout aligned with weekly pivot bias and 1-day trend filter, with volume confirmation (>1.5x 20-period average).
Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts.
Donchian(20) captures intermediate-term breakouts. 1-day EMA50 filter ensures alignment with daily trend.
Volume confirmation adds conviction. Designed for ~50-120 trades over 4 years (12-30/year) via tight confluence.
Works in bull markets via breakout continuation and in bear markets via trend-aligned short breakdowns.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use previous week's data (no look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan  # first week has no previous week
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot calculation
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # Support and resistance levels
    s1_1w = 2 * pivot_1w - prev_high_1w
    r1_1w = 2 * pivot_1w - prev_low_1w
    s2_1w = pivot_1w - (prev_high_1w - prev_low_1w)
    r2_1w = pivot_1w + (prev_high_1w - prev_low_1w)
    s3_1w = prev_low_1w - 2 * (prev_high_1w - pivot_1w)
    r3_1w = prev_high_1w + 2 * (pivot_1w - prev_low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    
    # Donchian(20) on 6h data
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        upper_donchian = highest_high[i]
        lower_donchian = lowest_low[i]
        
        if position == 0:
            # Long conditions: 1d uptrend + price breaks above Donchian upper + volume + above weekly pivot (bullish bias)
            long_condition = (close[i] > ema_trend and 
                            close[i] > upper_donchian and 
                            vol_regime[i] and 
                            close[i] > pivot)
            
            # Short conditions: 1d downtrend + price breaks below Donchian lower + volume + below weekly pivot (bearish bias)
            short_condition = (close[i] < ema_trend and 
                             close[i] < lower_donchian and 
                             vol_regime[i] and 
                             close[i] < pivot)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower or weekly S1 support
            if close[i] < lower_donchian or close[i] < s1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper or weekly R1 resistance
            if close[i] > upper_donchian or close[i] > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0