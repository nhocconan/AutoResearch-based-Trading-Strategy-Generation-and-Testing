#!/usr/bin/env python3
"""
6h_Weekly_Trend_Filter_With_1D_Pivot_Bounce
Hypothesis: Use weekly trend (price > weekly EMA34) to filter direction, then enter on 1D pivot (R1/S1) bounces with volume confirmation. In uptrend, buy near S1; in downtrend, sell near R1. Works in both bull/bear by aligning with weekly trend and capturing mean reversion at key daily levels. Targets 15-25 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA34
    close_weekly_series = pd.Series(close_weekly)
    ema34_weekly = close_weekly_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 6h timeframe (wait for weekly bar close)
    ema34_weekly_6h = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Get daily data for pivot points
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily pivot points (standard)
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L
    # S1 = 2*Pivot - H
    pivot_daily = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * pivot_daily - low_daily
    s1_daily = 2 * pivot_daily - high_daily
    
    # Align daily pivot levels to 6h timeframe (wait for daily bar close)
    r1_6h = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_6h = align_htf_to_ltf(prices, df_daily, s1_daily)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_weekly_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        uptrend = close[i] > ema34_weekly_6h[i]
        downtrend = close[i] < ema34_weekly_6h[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: in uptrend, price near S1 with volume confirmation
            if uptrend and abs(close[i] - s1_6h[i]) / s1_6h[i] < 0.005 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: in downtrend, price near R1 with volume confirmation
            elif downtrend and abs(close[i] - r1_6h[i]) / r1_6h[i] < 0.005 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses above R1 or trend changes
            if close[i] > r1_6h[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below S1 or trend changes
            if close[i] < s1_6h[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Trend_Filter_With_1D_Pivot_Bounce"
timeframe = "6h"
leverage = 1.0