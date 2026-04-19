#!/usr/bin/env python3
"""
1d_WeeklyPivot_Momentum_Breakout
Hypothesis: Weekly pivot levels (calculated from prior week's OHLC) act as strong support/resistance.
Breakout above weekly R1 or below weekly S1 with volume confirmation and ADX trend filter
captures momentum moves. Works in bull/bear via breakout logic and ADX filter to avoid whipsaw.
Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
"""

name = "1d_WeeklyPivot_Momentum_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
    
    dx = np.zeros_like(high)
    dx_sum = plus_di + minus_di
    mask = dx_sum > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / dx_sum[mask]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, S1, R2, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    _, weekly_r1, weekly_s1, _, _ = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to daily
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # ADX trend filter: only trade when ADX > 25
    adx = calculate_adx(high, low, close, 14)
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and trend
            if (close[i] > weekly_r1_aligned[i] and volume_confirm[i] and trend_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and trend
            elif (close[i] < weekly_s1_aligned[i] and volume_confirm[i] and trend_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below weekly S1 or trend weakens
            if (close[i] < weekly_s1_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above weekly R1 or trend weakens
            if (close[i] > weekly_r1_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals