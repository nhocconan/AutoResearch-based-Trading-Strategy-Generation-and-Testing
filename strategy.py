#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with Daily Trend Filter
Hypothesis: Trade reversals at Camarilla pivot levels (S3/R3) on 12h timeframe,
filtered by daily trend (EMA50) and volume confirmation. Works in bull/bear by
aligning with higher timeframe trend while capturing mean-reversion at key levels.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

name = "12h_Camarilla_Pivot_Reversal_1DTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Daily trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 12h bar
        # Using previous bar's OHLC for current bar's levels (no look-ahead)
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_ = ph - pl
        
        if range_ <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla levels
        s3 = pc - (range_ * 1.1 / 4)
        r3 = pc + (range_ * 1.1 / 4)
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long at S3 in daily uptrend with volume confirmation
            if daily_up and volume_confirm and low[i] <= s3 * 1.005:
                signals[i] = 0.25
                position = 1
            # Enter short at R3 in daily downtrend with volume confirmation
            elif daily_down and volume_confirm and high[i] >= r3 * 0.995:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches midpoint or trend changes
            midpoint = (s3 + r3) / 2
            if close[i] >= midpoint or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches midpoint or trend changes
            midpoint = (s3 + r3) / 2
            if close[i] <= midpoint or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals