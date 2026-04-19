#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Elder Ray index (Bull/Bear Power) and 1w trend filter.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and weekly trend up,
# Short when Bear Power < 0 and weekly trend down. Uses 1d EMA13 and 1w EMA21 for trend.
# Works in both bull and bear markets by capturing institutional buying/selling pressure.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_1d_ElderRay_1wTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA13 calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d timeframe
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Ensure enough data for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below EMA21
        weekly_uptrend = close_1w[-1] > ema21_1w[-1] if len(close_1w) > 0 else False
        weekly_downtrend = close_1w[-1] < ema21_1w[-1] if len(close_1w) > 0 else False
        
        # Get current weekly aligned values for decision making
        weekly_ema_aligned = ema21_1w_aligned[i]
        # Approximate weekly close price for trend check (use last known)
        weekly_close_approx = close[i]  # simplification for 6s chart
        
        if position == 0:
            # Long when Bull Power > 0 (buying pressure) and weekly uptrend
            if bull_power_aligned[i] > 0 and weekly_close_approx > weekly_ema_aligned:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 (selling pressure) and weekly downtrend
            elif bear_power_aligned[i] < 0 and weekly_close_approx < weekly_ema_aligned:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Bull Power turns negative (selling pressure appears)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Bear Power turns positive (buying pressure appears)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals