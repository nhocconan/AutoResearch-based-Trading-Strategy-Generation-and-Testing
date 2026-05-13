#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_Trend_Filter
Hypothesis: Weekly pivot levels provide strong support/resistance in ranging markets. 
Combining with 6h Donchian(20) breakouts and trend filters (EMA50) creates a robust strategy:
- Long when price breaks above Donchian high AND above weekly pivot (S1/S2) in uptrend
- Short when price breaks below Donchian low AND below weekly pivot (R1/R2) in downtrend
- Uses volume confirmation to avoid false breakouts
- Weekly pivot provides institutional reference points that work in both bull and bear markets
Target: 15-35 trades/year per symbol (60-140 total over 4 years)
"""

name = "6h_WeeklyPivot_Donchian_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # 6h Donchian Channel (20 periods)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_6h = close > ema_50
    downtrend_6h = close < ema_50
    
    # Weekly data for pivot points (calculate once)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, then support/resistance levels
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Get current values
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        ema50_val = ema_50[i]
        uptrend = uptrend_6h[i]
        downtrend = downtrend_6h[i]
        vol_ok = volume_conf[i]
        
        # Weekly pivot levels (aligned)
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        if position == 0:
            # LONG: Donchian breakout above weekly S1/S2 with uptrend and volume
            if (close[i] > donchian_high and 
                close[i] > s1_val and 
                uptrend and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown below weekly R1/R2 with downtrend and volume
            elif (close[i] < donchian_low and 
                  close[i] < r1_val and 
                  downtrend and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Donchian breakdown or trend reversal
            if close[i] < donchian_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Donchian breakout or trend reversal
            if close[i] > donchian_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals