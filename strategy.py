#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Breakout with 1w Trend Filter
Hypothesis: Weekly pivots provide strong support/resistance. Breakouts with volume confirmation
and 1-week trend alignment capture momentum moves in both bull and bear markets. Weekly trend
filter avoids counter-trend trades, reducing whipsaw in sideways markets.
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
    
    # Get weekly data for pivot and trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection: current volume > 2.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    # Price breakout conditions
    breakout_up = close > r1_aligned
    breakout_down = close < s1_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema50_1w_aligned[i]
        vol_ok = vol_spike[i]
        price = close[i]
        
        if position == 0:
            # Enter long on volume spike + breakout above R1 + uptrend
            if vol_ok and breakout_up[i] and price > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on volume spike + breakdown below S1 + downtrend
            elif vol_ok and breakout_down[i] and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on breakdown below S1 or trend change
            if close[i] < s1_aligned[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout above R1 or trend change
            if close[i] > r1_aligned[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Volume_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0