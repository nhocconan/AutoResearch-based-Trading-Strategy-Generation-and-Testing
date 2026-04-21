#!/usr/bin/env python3
"""
1d_1w_Momentum_Pivot_Breakout
Hypothesis: Weekly trend (EMA34) filters daily pivot breakouts (R1/S1). 
Only trade in direction of weekly trend: long when price > weekly EMA34, short when price < weekly EMA34.
Enter on break of daily R1/S1 with volume confirmation. Exit when price returns to daily pivot.
Designed for low frequency (<25 trades/year) to minimize fee drag. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Load daily data once for pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * pivot_daily - low_daily
    s1_daily = 2 * pivot_daily - high_daily
    
    # Align daily levels to 1d timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    pivot_daily_aligned = align_htf_to_ltf(prices, df_daily, pivot_daily)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or np.isnan(pivot_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_trend = ema34_weekly_aligned[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        pivot = pivot_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_ok = vol_current > 1.5 * vol_ma
        else:
            vol_ok = True  # insufficient data, allow trade
        
        if position == 0:
            # Long: price > weekly EMA34 and breaks above R1 with volume
            if price > weekly_trend and price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly EMA34 and breaks below S1 with volume
            elif price < weekly_trend and price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to daily pivot or breaks below S1
            if price <= pivot or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to daily pivot or breaks above R1
            if price >= pivot or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Momentum_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0