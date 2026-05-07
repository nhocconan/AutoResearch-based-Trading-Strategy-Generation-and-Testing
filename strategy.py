#!/usr/bin/env python3
name = "6h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader's pivot)
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily trend filter (using 1d EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20 periods) on 6h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume above 1.5x average
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: breakout above R2 with weekly uptrend (price above weekly pivot)
            if (close[i] > donchian_high[i] and 
                close[i] > r2_aligned[i] and 
                close[i] > pivot_aligned[i] and
                close[i] > ema_50_1d_aligned[i] and
                vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S2 with weekly downtrend (price below weekly pivot)
            elif (close[i] < donchian_low[i] and 
                  close[i] < s2_aligned[i] and 
                  close[i] < pivot_aligned[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or weekly pivot
            if close[i] < r1_aligned[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S1 or weekly pivot
            if close[i] > s1_aligned[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian breakout with weekly pivot levels and daily trend filter
# - Uses weekly pivot points (R2/S2) as significant support/resistance levels
# - Long when price breaks above Donchian(20) high AND R2 with volume confirmation in weekly uptrend
# - Short when price breaks below Donchian(20) low AND S2 with volume confirmation in weekly downtrend
# - Weekly pivot provides structure from higher timeframe (1w) that works in both bull/bear markets
# - Daily EMA(50) trend filter ensures alignment with intermediate trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to weekly pivot or corresponding support/resistance level
# - Position size 0.25 balances return potential with risk management
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Combines proven concepts: Donchian breakouts, pivot points, and trend filtering
# - Novel application: Weekly pivot levels as breakout thresholds on 6h timeframe