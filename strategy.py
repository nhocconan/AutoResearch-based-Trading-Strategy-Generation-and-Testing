#!/usr/bin/env python3
"""
6h_1w_1d_WeeklyPivot_Breakout_With_Volume
Hypothesis: Weekly pivot points (PP, R1/S1, R2/S2, R3/S3) identify key support/resistance zones.
Breakouts above R1 or below S1 with volume expansion capture institutional moves.
Weekly trend filter (price vs weekly EMA20) ensures trades align with higher timeframe momentum.
Works in bull markets (breakouts above R1/R2) and bear markets (breakdowns below S1/S2).
Target: 15-25 trades/year (~60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly trend filter: EMA20 on weekly close
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    
    # Get daily data for volume confirmation (more responsive than weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_ema_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume expansion condition
        volume_expansion = volume[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Long conditions:
        # 1. Breakout above R1 (first resistance)
        # 2. Price above weekly EMA20 (uptrend filter)
        # 3. Volume expansion
        breakout_r1 = close[i] > r1_aligned[i]
        price_above_ema = close[i] > weekly_ema_20_aligned[i]
        long_condition = breakout_r1 and price_above_ema and volume_expansion
        
        # Short conditions:
        # 1. Breakdown below S1 (first support)
        # 2. Price below weekly EMA20 (downtrend filter)
        # 3. Volume expansion
        breakdown_s1 = close[i] < s1_aligned[i]
        price_below_ema = close[i] < weekly_ema_20_aligned[i]
        short_condition = breakdown_s1 and price_below_ema and volume_expansion
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1w_1d_WeeklyPivot_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0