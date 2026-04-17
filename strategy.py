#!/usr/bin/env python3
"""
6h Elder Ray Power with Weekly Trend Filter
Long: Bull Power > 0 + weekly EMA(20) rising + price above weekly EMA(20)
Short: Bear Power < 0 + weekly EMA(20) falling + price below weekly EMA(20)
Exit: Opposite power crosses zero or price crosses weekly EMA
Uses Elder Ray (Bull/Bear Power) on 6h for entry timing, weekly EMA for trend filter.
Designed to capture trend continuations in both bull and bear markets by combining
momentum (Elder Ray) with higher timeframe trend.
Target: 80-150 total trades over 4 years (20-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_elder_ray(high, low, close, ema_length=13):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 26:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray on 6h
    bull_power, bear_power, ema_6h = calculate_elder_ray(high, low, close, ema_length=13)
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Align weekly EMA to 6h
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly EMA slope (1-period change) for trend filter
    ema_slope = np.diff(ema_20_1w_aligned, prepend=ema_20_1w_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need weekly EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_6h[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(ema_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 + price above weekly EMA + weekly EMA rising
            if bull_power[i] > 0 and price > weekly_ema and ema_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + price below weekly EMA + weekly EMA falling
            elif bear_power[i] < 0 and price < weekly_ema and ema_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power > 0 OR price crosses below weekly EMA
            if bear_power[i] > 0 or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power < 0 OR price crosses above weekly EMA
            if bull_power[i] < 0 or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_WeeklyTrend"
timeframe = "6h"
leverage = 1.0